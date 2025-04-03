import pandas as pd
import random
import unicodedata

# Load the original CSV
df = pd.read_csv('train.csv')

# Clean encoding in location column upfront
df['location'] = df['location'].str.encode('utf-8', errors='replace').str.decode('utf-8', errors='replace').fillna('')

# Load GeoNames data
geo_df = pd.read_csv('cities15000.txt', sep='\t', header=None, usecols=[1, 4, 5], names=['name', 'lat', 'lng'])
def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if c.isalnum()).lower()
city_coords = {normalize(row['name']): [row['lat'], row['lng']] for _, row in geo_df.iterrows()}
city_names = {normalize(row['name']): row['name'] for _, row in geo_df.iterrows()}  # Keep original names

# Function to generate random coordinates
def get_random_coords(base_coords=None):
    if base_coords:
        lat = base_coords[0] + random.uniform(-0.1, 0.1)  # ±0.1 degrees ~11km
        lng = base_coords[1] + random.uniform(-0.1, 0.1)
    else:
        lat = random.uniform(-90, 90)  # Global random
        lng = random.uniform(-180, 180)
    return [round(lat, 4), round(lng, 4)]

# Function to extract city from text or location
def extract_city(row):
    location = str(row['location']).strip()
    text = str(row['text']).strip()
    
    # Normalize full strings
    norm_location = normalize(location)
    norm_text = normalize(text)
    
    # 1. Exact or near-exact match for location
    for city in city_coords.keys():
        if norm_location == city or city in norm_location:
            return city
    
    # 2. Fallback: Check if full location is in text
    if norm_location and norm_location in norm_text:
        for city in city_coords.keys():
            if city in norm_location:
                return city
    
    # 3. Last resort: Check text for full city names
    for city in city_coords.keys():
        if city in norm_text:
            return city
    
    return None

# Add coordinates and update location
def update_row(row):
    matched_city = extract_city(row)
    if matched_city:
        row['location'] = city_names[matched_city]  # Use original city name (e.g., "Hawai‘i Kai")
        row['coordinates'] = get_random_coords(city_coords[matched_city])
    else:
        row['coordinates'] = get_random_coords(None)  # Random if no match
    return row

df = df.apply(update_row, axis=1)

# Save the updated CSV
df.to_csv('train_with_coords.csv', index=False)
print(f"New CSV 'train_with_coords.csv' created with {len(city_coords)} cities mapped!")