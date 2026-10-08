import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import os

class KarnatakaPostalWeather:
    def __init__(self, openweather_api_key, postal_csv_path):
        self.api_key = openweather_api_key
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'PostalWeather/1.0'})
        
        # Load and preprocess postal data
        self.postal_data = self._load_postal_data(postal_csv_path)
        
        # District center coordinates for fallback
        self.district_centers = {
            "BAGALKOT": (16.1867, 75.6961),
            "BANGALORE URBAN": (12.9716, 77.5946),
            "BELGAUM": (15.8497, 74.4977),
            # Add all other districts from your data
            "BIJAPUR": (16.8240, 75.7154),
            "CHIKKAMAGALURU": (13.3161, 75.7720),
            # ... include all 31 districts
        }

    def _load_postal_data(self, csv_path):
        """Load and preprocess postal office data"""
        try:
            # Read CSV, skip initial rows if they contain metadata
            df = pd.read_csv(csv_path, skiprows=2, header=None)
            
            # Assign proper column names based on your format
            df.columns = [
                'office_name', 'office_status', 'pincode', 'telephone',
                'taluk', 'district', 'state', 'postal_division',
                'postal_region', 'postal_circle'
            ]
            
            # Clean data
            df['pincode'] = df['pincode'].astype(str).str.strip()
            df['district'] = df['district'].str.strip().str.upper()
            
            # Remove duplicates (same pincode in same district)
            df = df.drop_duplicates(['pincode', 'district'])
            
            return df
        except Exception as e:
            raise ValueError(f"Error processing postal data: {str(e)}")

    def _get_coordinates(self, pincode, district):
        """Get coordinates with optimized fallback logic"""
        try:
            # Try with district first for better accuracy
            if pd.notna(district):
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'postalcode': pincode,
                    'country': 'India',
                    'county': district,
                    'state': 'Karnataka',
                    'format': 'json',
                    'limit': 1
                }
                response = self.session.get(url, params=params)
                data = response.json()
                if data:
                    return float(data[0]['lat']), float(data[0]['lon'])
        except:
            pass
        
        # Fallback to district center if available
        if district in self.district_centers:
            return self.district_centers[district]
        
        # Final fallback to Bangalore
        return (12.9716, 77.5946)

    def fetch_weather_for_postal_data(self, output_path=None):
        """Fetch weather for all postal offices in the CSV"""
        weather_data = []
        unique_pincodes = self.postal_data[['pincode', 'district']].drop_duplicates()
        
        # Process each unique pincode-district pair
        for _, row in tqdm(unique_pincodes.iterrows(), total=len(unique_pincodes), desc="Processing Pincodes"):
            pincode = row['pincode']
            district = row['district']
            
            try:
                lat, lon = self._get_coordinates(pincode, district)
                
                # Get weather data
                weather_url = "https://api.openweathermap.org/data/2.5/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'units': 'metric',
                    'appid': self.api_key
                }
                response = self.session.get(weather_url, params=params)
                weather = response.json()

                # Get air quality
                aq_url = "http://api.openweathermap.org/data/2.5/air_pollution"
                aq_response = self.session.get(aq_url, params=params)
                aq_data = aq_response.json()

                # Create weather record
                weather_record = {
                    'pincode': pincode,
                    'district': district,
                    'latitude': lat,
                    'longitude': lon,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'temperature': weather['main']['temp'],
                    'temp_min': weather['main']['temp_min'],
                    'temp_max': weather['main']['temp_max'],
                    'humidity': weather['main']['humidity'],
                    'rainfall': weather.get('rain', {}).get('1h', 0),
                    'air_quality': aq_data['list'][0]['main']['aqi'],
                    'conditions': weather['weather'][0]['description'],
                    'wind_speed': weather['wind']['speed']
                }
                weather_data.append(weather_record)
                
                # Respect API rate limits
                time.sleep(1.2)
                
            except Exception as e:
                print(f"\nError processing {pincode} ({district}): {str(e)}")
                continue
        
        # Merge weather data with original postal data
        weather_df = pd.DataFrame(weather_data)
        merged_df = pd.merge(
            self.postal_data,
            weather_df,
            on=['pincode', 'district'],
            how='left'
        )
        
        # Save results
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            output_path = f"karnataka_postal_weather_{timestamp}.csv"
        
        merged_df.to_csv(output_path, index=False)
        return merged_df, output_path

if __name__ == "__main__":
    # Configuration
    API_KEY = "a710553d9d99f30326a82d432265cf36"  # Replace with your actual API key
    POSTAL_CSV_PATH = "your_postal_data.csv"  # Path to your postal office CSV
    
    # Initialize and process
    processor = KarnatakaPostalWeather(
        openweather_api_key=API_KEY,
        postal_csv_path=POSTAL_CSV_PATH
    )
    
    print("Starting weather data collection for postal offices...")
    try:
        result_df, output_file = processor.fetch_weather_for_postal_data()
        
        print("\nProcessing complete!")
        print(f"Total records processed: {len(result_df)}")
        print(f"Records with weather data: {len(result_df.dropna(subset=['temperature']))}")
        print(f"Output saved to: {output_file}")
        
        print("\nSample of merged data:")
        print(result_df.sample(3)[[
            'office_name', 'pincode', 'district', 
            'temperature', 'conditions', 'air_quality'
        ]])
    except Exception as e:
        print(f"\nFailed to complete processing: {str(e)}")