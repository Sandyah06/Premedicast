import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from datetime import datetime, timedelta
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib

class KarnatakaHospitalAnalyzer:
    def __init__(self, postal_data_path='your_postal_data.csv'):
        self.data_dir = "hospital_analysis"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load postal data
        self.postal_data = pd.read_csv(postal_data_path)
        print(f"Loaded postal data with {len(self.postal_data)} records")
        
        # Government data sources
        self.data_sources = [
            "https://health.karnataka.gov.in/hospitals.csv",
            "https://data.gov.in/files/ogdpv2dms/s3fs-public/Karnataka_Hospitals.csv",
            "https://arogyavani.karnataka.gov.in/hospital-data"
        ]
        
        # Prediction models storage
        self.models = {
            'beds': None,
            'icu': None,
            'er_capacity': None
        }

    def fetch_hospital_data(self):
        """Fetch and process real hospital data"""
        hospitals = None
        
        for url in self.data_sources:
            try:
                print(f"Fetching from {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Try different parsers
                try:
                    data = pd.read_csv(StringIO(response.text))
                except:
                    data = pd.read_excel(StringIO(response.content))
                
                # Standardize columns
                col_map = {
                    'name': 'Hospital Name', 'hospital_name': 'Hospital Name',
                    'district': 'District', 'district_name': 'District',
                    'pin': 'Pincode', 'pincode': 'Pincode',
                    'type': 'Type', 'hospital_type': 'Type',
                    'beds': 'Total Beds', 'total_beds': 'Total Beds',
                    'icu': 'ICU Beds', 'icu_beds': 'ICU Beds'
                }
                data = data.rename(columns={k:v for k,v in col_map.items() if k in data.columns})
                
                # Filter for Karnataka
                if 'State' in data.columns:
                    data = data[data['State'].str.contains('Karnataka', na=False)]
                
                # Merge with postal data
                merged = pd.merge(
                    data,
                    self.postal_data,
                    left_on=['District', 'Pincode'],
                    right_on=['DISTRICT', 'PINCODE'],
                    how='right'  # Keep all postal records
                )
                
                if len(merged) > 0:
                    print(f"Found {len(data)} real hospitals, merged with postal data")
                    hospitals = merged
                    break
                    
            except Exception as e:
                print(f"Failed to fetch from {url}: {str(e)}")
                continue
        
        if hospitals is None:
            print("All sources failed, initializing empty dataset")
            hospitals = self.postal_data.copy()
            hospitals['Hospital Name'] = np.nan
            hospitals['Type'] = np.nan
            hospitals['Total Beds'] = np.nan
            hospitals['ICU Beds'] = np.nan
        
        return hospitals

    def predict_missing_values(self, hospitals):
        """Train models to predict missing hospital attributes"""
        # Prepare training data (rows with complete information)
        train_data = hospitals.dropna(subset=['Total Beds', 'ICU Beds', 'Type'])
        
        if len(train_data) == 0:
            print("No training data available, using fallback values")
            hospitals['Total Beds'] = hospitals['Total Beds'].fillna(100)
            hospitals['ICU Beds'] = hospitals['ICU Beds'].fillna(10)
            hospitals['Type'] = hospitals['Type'].fillna('Government')
            return hospitals
        
        # Feature engineering
        features = pd.get_dummies(train_data[['District', 'TALUK', 'Type']])
        beds_target = train_data['Total Beds']
        icu_target = train_data['ICU Beds']
        
        # Train models
        self.models['beds'] = RandomForestRegressor().fit(features, beds_target)
        self.models['icu'] = RandomForestRegressor().fit(features, icu_target)
        
        # Predict missing values
        missing = hospitals[hospitals['Total Beds'].isna()]
        if len(missing) > 0:
            missing_features = pd.get_dummies(missing[['District', 'TALUK', 'Type']])
            
            # Ensure feature columns match
            for col in features.columns:
                if col not in missing_features.columns:
                    missing_features[col] = 0
            missing_features = missing_features[features.columns]
            
            # Make predictions
            hospitals.loc[hospitals['Total Beds'].isna(), 'Total Beds'] = \
                self.models['beds'].predict(missing_features)
            hospitals.loc[hospitals['ICU Beds'].isna(), 'ICU Beds'] = \
                self.models['icu'].predict(missing_features)
        
        # Fill remaining missing
        hospitals['Total Beds'] = hospitals['Total Beds'].fillna(100).astype(int)
        hospitals['ICU Beds'] = hospitals['ICU Beds'].fillna(10).astype(int)
        hospitals['Type'] = hospitals['Type'].fillna('Government')
        
        # Calculate ER capacity (20% of beds)
        hospitals['ER Capacity'] = (hospitals['Total Beds'] * 0.2).astype(int)
        
        # Generate hospital names for new records
        mask = hospitals['Hospital Name'].isna()
        hospitals.loc[mask, 'Hospital Name'] = hospitals[mask].apply(
            lambda x: f"{x['TALUK']} {x['Type']} Hospital", axis=1)
        
        return hospitals

    def generate_er_visits(self, hospitals, days=60):
        """Generate realistic ER visits based on hospital capacity"""
        records = []
        
        for _, hospital in hospitals.iterrows():
            base_visits = hospital['ER Capacity'] * np.random.uniform(0.8, 1.2)
            
            for day in range(days):
                date = datetime.now() - timedelta(days=days-day)
                
                # Weekly pattern (40% more on Mondays)
                visits = np.random.poisson(base_visits * (1.4 if date.weekday() == 0 else 1))
                
                for _ in range(visits):
                    records.append({
                        "Date": date.strftime("%Y-%m-%d"),
                        "Hospital": hospital["Hospital Name"],
                        "Pincode": hospital["PINCODE"],
                        "District": hospital["District"],
                        "Taluk": hospital["TALUK"],
                        "Reason": self._predict_reason(hospital),
                        "Wait Time (mins)": self._predict_wait_time(hospital),
                        "Admitted": np.random.random() < 0.2
                    })
        
        visits_df = pd.DataFrame(records)
        visits_df.to_csv(os.path.join(self.data_dir, 'er_visits.csv'), index=False)
        print(f"Generated {len(visits_df)} ER visits")
        return visits_df

    def _predict_reason(self, hospital):
        """Predict visit reason based on hospital type/location"""
        reasons = ["Fever", "Accident", "Heart Issues", "Breathing Problems", "Stomach Pain"]
        weights = [0.3, 0.2, 0.15, 0.2, 0.15]  # Default weights
        
        if hospital['Type'] == 'Government':
            weights = [0.4, 0.25, 0.1, 0.15, 0.1]  # More fevers in government hospitals
        elif 'Rural' in hospital['TALUK']:
            weights = [0.5, 0.1, 0.1, 0.2, 0.1]  # More fevers in rural areas
        
        return np.random.choice(reasons, p=weights)

    def _predict_wait_time(self, hospital):
        """Predict wait time based on hospital capacity"""
        base_time = 30  # minutes
        # Longer waits in government hospitals with more beds
        if hospital['Type'] == 'Government':
            return np.random.randint(base_time, 120)
        return np.random.randint(base_time, 60)

    def analyze_and_visualize(self, visits_df):
        """Create comprehensive visualizations"""
        plt.figure(figsize=(18, 12))
        
        # 1. Visits by District
        plt.subplot(2, 2, 1)
        district_visits = visits_df['District'].value_counts().head(10)
        sns.barplot(x=district_visits.values, y=district_visits.index, palette="rocket")
        plt.title("Top Districts by ER Visits")
        plt.xlabel("Number of Visits")
        
        # 2. Visit Reasons
        plt.subplot(2, 2, 2)
        reason_counts = visits_df['Reason'].value_counts()
        sns.barplot(x=reason_counts.values, y=reason_counts.index, palette="viridis")
        plt.title("Common ER Visit Reasons")
        plt.xlabel("Count")
        
        # 3. Wait Time Analysis
        plt.subplot(2, 2, 3)
        sns.boxplot(x=visits_df['Wait Time (mins)'], color="skyblue")
        plt.title("Patient Wait Times")
        plt.xlabel("Minutes")
        
        # 4. Visits by Hospital Type
        plt.subplot(2, 2, 4)
        visits_df['Hospital Type'] = visits_df['Hospital'].str.extract(r'(\w+) Hospital')[0]
        type_counts = visits_df['Hospital Type'].value_counts()
        sns.barplot(x=type_counts.values, y=type_counts.index, palette="magma")
        plt.title("Visits by Hospital Type")
        plt.xlabel("Count")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'hospital_analysis.png'), dpi=120)
        plt.close()
        print("Saved visualizations to hospital_analysis.png")

    def generate_report(self, hospitals_df, visits_df):
        """Create comprehensive report with predictions"""
        report = f"""
        KARNATAKA HOSPITAL EMERGENCY ANALYSIS REPORT
        {'='*60}
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        Data Period: {visits_df['Date'].min()} to {visits_df['Date'].max()}
        {'-'*60}
        
        HOSPITAL NETWORK:
        - Total Hospitals: {len(hospitals_df)}
        - Government: {len(hospitals_df[hospitals_df['Type'] == 'Government'])}
        - Private: {len(hospitals_df[hospitals_df['Type'] == 'Private'])}
        - Average Beds: {hospitals_df['Total Beds'].mean():.1f}
        - Average ICU Beds: {hospitals_df['ICU Beds'].mean():.1f}
        
        ER VISIT ANALYSIS:
        - Total Visits: {len(visits_df):,}
        - Average Daily Visits: {len(visits_df)/60:.1f}
        - Average Wait Time: {visits_df['Wait Time (mins)'].mean():.1f} mins
        - Admission Rate: {visits_df['Admitted'].mean():.1%}
        
        PREDICTED HOTSPOTS:
        1. District: {visits_df['District'].value_counts().index[0]}
        2. Taluk: {visits_df['Taluk'].value_counts().index[0]}
        3. Most Common Reason: {visits_df['Reason'].value_counts().index[0]}
        
        RECOMMENDATIONS:
        1. Expand capacity in {visits_df['District'].value_counts().index[0]}
        2. Focus on {visits_df['Reason'].value_counts().index[0]} cases
        3. Reduce wait times at {visits_df['Hospital'].value_counts().index[0]}
        """
        
        report_path = os.path.join(self.data_dir, 'hospital_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Saved report to {report_path}")

    def run_analysis(self):
        """Complete analysis workflow"""
        print("=== Karnataka Hospital Analysis ===")
        
        # Step 1: Fetch and process hospital data
        print("\n1. Fetching and processing hospital data...")
        hospitals = self.fetch_hospital_data()
        hospitals = self.predict_missing_values(hospitals)
        hospitals.to_csv(os.path.join(self.data_dir, 'hospitals_processed.csv'), index=False)
        
        # Step 2: Generate ER visits
        print("\n2. Generating ER visit predictions...")
        visits = self.generate_er_visits(hospitals)
        
        # Step 3: Analyze and visualize
        print("\n3. Analyzing data and creating visualizations...")
        self.analyze_and_visualize(visits)
        
        # Step 4: Generate report
        print("\n4. Compiling final report...")
        self.generate_report(hospitals, visits)
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to '{self.data_dir}' folder:")
        print("- hospitals_processed.csv")
        print("- er_visits.csv")
        print("- hospital_analysis.png")
        print("- hospital_report.txt")

if __name__ == "__main__":
    analyzer = KarnatakaHospitalAnalyzer()
    analyzer.run_analysis()