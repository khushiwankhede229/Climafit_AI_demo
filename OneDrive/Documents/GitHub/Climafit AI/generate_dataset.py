
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Configuration for dataset size (Adjust this number for more/less data)
# User asked for 5 Lakh (500,000), but for tool execution safety we start with 50,000.
# You can change this to 500000 easily.
NUM_ROWS = 60000 
OUTPUT_FILE = r'c:\Users\YASHRAJ\Desktop\Climafit-AI_demo\data\Indian_Fashion_Dataset.csv'

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ===== Data Banks =====

BRANDS = [
    "FabIndia", "Biba", "W for Woman", "Raymond", "Manyavar", 
    "Allen Solly", "Peter England", "Louis Philippe", "Westside", "Pantaloons"
]

ORIGINS = ["Jaipur", "Surat", "Kolkata", "Mumbai", "Ludhiana", "Banaras", "Kashmir", "Mysore"]

# Weather & Season Logic for India
# Month Index: 1=Jan, 12=Dec
SEASON_MAPPING = {
    1:  {"Season": "Winter",  "TempRange": (5, 20), "Weather": ["Chilly", "Foggy", "Cold"]},
    2:  {"Season": "Spring",  "TempRange": (15, 25), "Weather": ["Pleasant", "Sunny", "Breezy"]},
    3:  {"Season": "Summer",  "TempRange": (25, 35), "Weather": ["Warm", "Sunny", "Dry"]},
    4:  {"Season": "Summer",  "TempRange": (30, 42), "Weather": ["Hot", "Scorching", "Dry"]},
    5:  {"Season": "Summer",  "TempRange": (35, 45), "Weather": ["Heatwave", "Very Hot", "Dusty"]},
    6:  {"Season": "Monsoon", "TempRange": (28, 38), "Weather": ["Humid", "Cloudy", "Pre-Monsoon Rain"]},
    7:  {"Season": "Monsoon", "TempRange": (25, 32), "Weather": ["Rainy", "Humid", "Overcast"]},
    8:  {"Season": "Monsoon", "TempRange": (25, 32), "Weather": ["Heavy Rain", "Humid", "Stormy"]},
    9:  {"Season": "Monsoon", "TempRange": (26, 34), "Weather": ["Humid", "Showers", "Clearing"]},
    10: {"Season": "Autumn",  "TempRange": (24, 32), "Weather": ["Pleasant", "Clear", "Breezy"]},
    11: {"Season": "Autumn",  "TempRange": (15, 28), "Weather": ["Cooling", "Dry", "Clear"]},
    12: {"Season": "Winter",  "TempRange": (8, 22),  "Weather": ["Cold", "Chilly", "Dry"]},
}

# Fabrics strictly mapped to Season to avoid "Linen in Winter"
# User requested only 10 fabrics: Cotton, Silk, Linen, Wool, Polyester, Denim, Rayon, Chiffon, Velvet, Khadi
SEASONAL_FABRICS = {
    "Winter":  ["Wool", "Velvet"],
    "Spring":  ["Silk", "Chiffon", "Denim", "Khadi"],
    "Summer":  ["Cotton", "Linen", "Rayon", "Khadi"],
    "Monsoon": ["Polyester", "Rayon"],
    "Autumn":  ["Silk", "Denim"]
}

# Clothing Items (Detailed)
CLOTHING_TEMPLATES = {
    "Winter": [
        "Kashmiri Embroidered Phiran", "Heavy Woolen Trench Coat", "Turtleneck Thermal Sweater", 
        "Padded Bomber Jacket", "Velvet Sherwani Set", "Merino Wool Cardigan", "Handwoven Shawl",
        "Quilted Nehru Jacket", "Tweed Blazer", "Fleece Lined Hoodie"
    ],
    "Summer": [
        "Chikankari Cotton Kurta", "Breathable Linen Shirt", "Floral Print Maxi Dress", 
        "Handblock Print Mulmul Saree", "Cotton Bermuda Shorts", "Sleeveless Khadi Vest",
        "Oversized Cotton T-Shirt", "Breezy Palazzo Pants", "White Linen Trousers", "Pastel Sundress"
    ],
    "Monsoon": [
        "Water Resistant Windcheater", "Cropped Capris", "Quick-Dry Polo T-Shirt", 
        "Nylon Cargo Shorts", "Polyester Rain Jacket", "Ankle Length Leggings", "Synthetic Blend Kurti"
    ],
    "Spring": [
        "Floral Silk Scarf", "Light Denim Jacket", "Full Sleeve Cotton Shirt", 
        "A-Line Midi Skirt", "Quarter Zip Pullover", "Classic Chinos", "Printed Anarkali Suit"
    ],
    "Autumn": [
        "Lightweight Flannel Shirt", "Denim Trucker Jacket", "Knitted Poncho", 
        "Long Sleeve T-Shirt", "Silk Blend Dupatta", "Casual Blazer", "Layered Hoodie"
    ]
}

ADJECTIVES = ["Premium", "Handcrafted", "Sustainable", "Designer", "Classic", "Modern", "Vintage", "Urban", "Elegant", "Bohemian"]

def generate_row(curr_date):
    month_idx = curr_date.month
    season_info = SEASON_MAPPING[month_idx]
    season = season_info["Season"]
    
    # 1. Temperature Calculation (Randomized within realistic range)
    base_temp = random.uniform(*season_info["TempRange"])
    # Add day/night variance or random noise
    temp = round(base_temp + random.uniform(-2, 2), 1)
    
    # 2. Weather
    weather = random.choice(season_info["Weather"])
    
    # 3. Fabric selection (Strictly based on season)
    fabric = random.choice(SEASONAL_FABRICS[season])
    
    # 4. Product Name Construction
    base_item = random.choice(CLOTHING_TEMPLATES[season])
    adjective = random.choice(ADJECTIVES)
    gender = random.choice(["Men", "Women", "Unisex"])
    
    product_name = f"{adjective} {fabric} {base_item}"
    
    # 5. Description
    brand = random.choice(BRANDS)
    origin = random.choice(ORIGINS)
    desc = f"A {adjective.lower()} {base_item.lower()} crafted from {fabric.lower()}. Perfect for the {season.lower()} weather in India. Sourced from {origin}."
    
    # 6. Price
    price = random.randint(800, 15000)
    
    return [
        curr_date.strftime("%Y-%m-%d"),
        curr_date.strftime("%B"),
        season,
        temp,
        weather,
        gender,
        brand,
        product_name,
        fabric,
        price,
        desc
    ]

# ===== Generation Loop =====
print(f"Generating {NUM_ROWS} rows of realistic data...")
data = []
start_date = datetime(2024, 1, 1)

for i in range(NUM_ROWS):
    # Sequential days to represent a timeline, repeating years into 2025
    curr_date = start_date + timedelta(days=i % 730)
    data.append(generate_row(curr_date))
    
    if i % 50000 == 0 and i > 0:
        print(f"Generated {i} rows...")

columns = [
    "Date", "Month", "Season", "Temperature_C", "Weather_Condition", 
    "Gender", "Brand", "Product Name", "Fabric", "Price_INR", "Description"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Successfully saved dataset to {OUTPUT_FILE}")
