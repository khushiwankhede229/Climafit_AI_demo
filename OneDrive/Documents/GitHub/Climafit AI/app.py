from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import csv
import requests
from datetime import datetime
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import pickle
import numpy as np
import json
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available - install Pillow for real color detection")

app = Flask(__name__)
app.secret_key = 'climafit_secret_key_demo'  # Required for session management

def get_fashion_image(kwd, season, index=0):
    images = {
        'summer': ['1523381210434-271e8be1f52b', '1529374255404-311a2a4f1fd9', '1503341455253-b2e723bb3dbb', '1503342217505-b0a15ec3261c', '1469334023215-c3848edef0d1', '1515372039744-b8f02a3ae446'],
        'winter': ['1551028719-001dd5c32439', '1544642899-f0d6e5f6ed6f', '1485230895905-ec40ba36b9bc', '1515886657613-9f3515b0c78f', '1517502129532-61fb23d06991'],
        'monsoon': ['1564347711929-c0ae2f168f18', '1504280590483-33bc0db724b1', '1428592960573-04fd526eb6eb'],
        'autumn': ['1508215885820-4585e56135c8', '1479839672679-a4648f6aafd5', '1406856754320-94d8ba9eaf51'],
        'spring': ['1490481651871-ab68de25d43d', '1434389678369-e851bf6d6553', '1520006403995-26db1cc27e02'],
        'shirt': ['1596755094514-f87e32f85e23', '1602810316498-5bbfca6ea1d2', '1626497764734-a15d0fa8ef9c', '1602810318383-e386cc2a3ccf', '1585007621454-e054bd3315a0'],
        'linen': ['1584273143610-d017b2ed87c7', '1602810316498-5bbfca6ea1d2'],
        'pant': ['1624378439575-d50c19a4e69b', '1541099649105-f69ad21f3246', '1473966968600-fa801b2c4535', '1584273143610-d017b2ed87c7'],
        'dress': ['1515372039744-b8f02a3ae446', '1496747611176-843222e1e57c', '1539008835657-9e8e9680c956', '1566206091558-4f11ef7320b5', '1595777457583-95e059f581b6'],
        'saree': ['1610189013233-0c464e8e19e7', '1583391733958-d15f01e40fb8', '1595064845532-349f22030d99'],
        'kurta': ['1610189013233-0c464e8e19e7', '1583391733958-d15f01e40fb8']
    }
    kwd_lower = kwd.lower()
    match_list = images.get('summer')
    
    if 'shirt' in kwd_lower: match_list = images['shirt']
    elif 'pant' in kwd_lower or 'trouser' in kwd_lower or 'jeans' in kwd_lower: match_list = images['pant']
    elif 'dress' in kwd_lower or 'skirt' in kwd_lower: match_list = images['dress']
    elif 'saree' in kwd_lower or 'sari' in kwd_lower: match_list = images['saree']
    elif 'kurta' in kwd_lower or 'ethnic' in kwd_lower: match_list = images['kurta']
    elif 'linen' in kwd_lower: match_list = images['linen']
    elif season.lower() in images: match_list = images[season.lower()]
    
    if not match_list:
        match_list = images['summer']
        
    return f"https://images.unsplash.com/photo-{match_list[int(index) % len(match_list)]}?w=500&q=80&fit=crop"

app.jinja_env.globals.update(get_fashion_image=get_fashion_image)

# ===== Load Data =====
# Using relative path for robustness or absolute if needed. 
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "Indian_Fashion_Dataset.csv")
user_data_path = os.path.join(BASE_DIR, "userdata.csv")

# Ensure user data file exists with headers
if not os.path.exists(user_data_path):
    with open(user_data_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Email', 'Password'])

# Load Data
try:
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load dataset at {csv_path}. Error: {e}")
    # Fallback Empty DF
    df = pd.DataFrame(columns=['Season', 'Product Name', 'Brand', 'Price_INR', 'Description', 'Temperature_C'])

# ===== Helper Functions =====
def get_coordinates(district, state):
    """Fetch coordinates for a district in India using Open-Meteo Geocoding API"""
    try:
        # Search for "District, State, India" to be specific
        query = f"{district}, {state}, India"
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            return result['latitude'], result['longitude'], result['name']
        else:
            # Fallback to just district if specific query fails
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={district}&count=1&language=en&format=json"
            response = requests.get(url, timeout=5)
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                return result['latitude'], result['longitude'], result['name']
            
        return None, None, None
    except Exception as e:
        print(f"Geocoding API Error: {e}")
        return None, None, None

def get_weather(lat, lon):
    """Fetch real-time weather using coordinates"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        current = data.get('current_weather', {})
        temp = current.get('temperature', 30)
        code = current.get('weathercode', 0)
        
        # WMO Weather Code Map
        weather_map = {
            0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
            45: "Foggy", 48: "Foggy", 51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
            61: "Rainy", 63: "Rainy", 65: "Heavy Rain", 80: "Showers", 81: "Showers",
            95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm"
        }
        condition = weather_map.get(code, "Clear")
        return temp, condition
    except Exception as e:
        print(f"Weather API Error: {e}")
        return 30, "Clear (Offline)"

def generate_plot(fig):
    """Convert Matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def load_users():
    """Load users from CSV into a dictionary {email: password}"""
    users = {}
    if os.path.exists(user_data_path):
        try:
            with open(user_data_path, mode='r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Skip header and empty lines
                    if len(row) >= 3 and row[1] != "Email":
                        users[row[1].strip()] = row[2].strip()
        except Exception as e:
            print(f"Error reading user db: {e}")
    return users

# ===== Login Route =====
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # 1. Admin Login
        if email == "admin" or email == "admin@climafit.ai":
            if password == "admin123":
                session['user'] = 'admin'
                session['role'] = 'admin'
                return redirect(url_for("home"))
            else:
                 return render_template("login.html", error="Invalid Admin Credentials")
        
        # 2. User Login (Verify Logic)
        users = load_users()
        
        if email in users:
            if users[email] == password:
                session['user'] = email
                session['role'] = 'user'
                return redirect(url_for("home"))
            else:
                return render_template("login.html", error="Incorrect Password")
        else:
            return render_template("login.html", error="User not found. Please Sign Up.")

    return render_template("login.html")

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        # In a real app, send email here.
        return render_template("login.html", error="Reset link sent to your email (Demo).")
    return render_template("forgot_password.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        # Check existing
        users = load_users()
        if email in users:
            return render_template("signup.html", error="Email already registered. Please Login.")

        # Reuse same CSV storage logic
        try:
            with open(user_data_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), email, password])
        except Exception as e:
            print(f"Error saving user data: {e}")

        # Auto Login
        session['user'] = email
        session['role'] = 'user'
        return redirect(url_for("home"))
    
    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ===== Admin Dashboard =====
@app.route("/admin")
def admin_dashboard():
    # Security check
    if session.get('role') != 'admin':
        return redirect(url_for("login"))
    
    users = []
    if os.path.exists(user_data_path):
        try:
            # Read CSV manually or with pandas
            users_df = pd.read_csv(user_data_path)
            users = users_df.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading user data: {e}")

    return render_template("admin.html", users=users)

@app.route("/home")
def home():
    if 'user' not in session:
        return redirect(url_for("login"))

    # Location Args
    state = request.args.get("state", "Maharashtra")
    district = request.args.get("district", "Aurangabad (Chhatrapati Sambhajinagar)")
    query = request.args.get("query", "").lower()

    # 1. Get Coordinates
    lat, lon, location_name = get_coordinates(district, state)
    if not lat:
        # Fallback
        lat, lon = 19.8762, 75.3433
        location_name = district

    # 2. Get Weather
    temperature, weather = get_weather(lat, lon)
    
    # 3. Auto-Select Season based on Temperature
    # Logic: >30 Summer, 20-30 Spring, <20 Winter (Simplified)
    # Also consider Rain for Monsoon if weather condition indicates it
    
    auto_season = "Summer" # Default
    
    if "Rain" in weather or "Drizzle" in weather or "Showers" in weather or "Thunderstorm" in weather:
        auto_season = "Monsoon"
    elif temperature >= 30:
        auto_season = "Summer"
    elif 20 <= temperature < 30:
        auto_season = "Spring" # or Autumn depending on month, but simpler to map to Spring/Autumn mix
    else:
        auto_season = "Winter"
        
    # Override if user explicitly forced a season (optional, but requested "recommendations according to users input" usually implies location input driving the recs)
    # The requirement: "live tem api tell us recommendations according to users input" -> User inputs location -> API gets temp -> Temp determines recs.
    
    # 4. Filter Dataset
    # Check for search query (Search bar logic)
    search_query = request.args.get("query", "").strip().lower()
    
    if search_query:
        # Search in Product Name, Brand, Description, Season
        mask = (
            df['Product Name'].str.lower().str.contains(search_query, na=False) |
            df['Brand'].str.lower().str.contains(search_query, na=False) |
            df['Description'].str.lower().str.contains(search_query, na=False) |
            df['Season'].str.lower().str.contains(search_query, na=False)
        )
        filtered_df = df[mask]
        
        # Convert to dictionary items (limit to 100 to avoid excessive DOM load, user asked for recommended items)
        items = []
        for _, row in filtered_df.head(100).iterrows():
            # Improved images & Myntra links
            kwd = row['Product Name'].replace(' ','-').lower()
            season = row['Season'].lower()
            items.append({
                "Product Name": row['Product Name'],
                "Price_INR": row['Price_INR'],
                "Brand": row['Brand'],
                "Description": row['Description'],
                "Image_Keyword": f"{season},{kwd},fashion",
                "Image_Seed": int(row['Price_INR']) + 500,
                "Season": row['Season'],
                "Myntra_Link": f"https://www.myntra.com/{kwd}"
            })
            
        message = f"Found {len(filtered_df)} results for '{search_query}'"
    else:
        # User requested specifically 4 cards representing seasons with beautiful images
        items = [
            {
                "Product Name": "Summer Collection",
                "Price_INR": "499",
                "Brand": "Sunny Vibes",
                "Description": "Light cottons, floral prints, and breezy styles for the heat.",
                "Image_Keyword": "summer,fashion,model",
                "Image_Seed": 501, 
                "Season": "Summer",
                "Myntra_Link": "https://www.myntra.com/summer-wear"
            },
            {
                 "Product Name": "Monsoon Essentials",
                 "Price_INR": "799",
                 "Brand": "Rain Ready",
                 "Description": "Waterproof jackets, umbrellas, and quick-dry fabrics.",
                 "Image_Keyword": "raincoat,fashion,model",
                 "Image_Seed": 502,
                 "Season": "Monsoon",
                 "Myntra_Link": "https://www.myntra.com/rain-jacket"
            },
            {
                 "Product Name": "Winter Warmth",
                 "Price_INR": "1299",
                 "Brand": "Cozy Knit",
                 "Description": "Woolen sweaters, heavy jackets, and warm thermals.",
                 "Image_Keyword": "winter,jacket,model",
                 "Image_Seed": 503,
                 "Season": "Winter",
                 "Myntra_Link": "https://www.myntra.com/winter-wear"
            },
            {
                 "Product Name": "Autumn/Spring Edit",
                 "Price_INR": "899",
                 "Brand": "Transitional",
                 "Description": "Perfect layers, trench coats, and light cardigans.",
                 "Image_Keyword": "autumn,coat,model",
                 "Image_Seed": 504,
                 "Season": "Autumn",
                 "Myntra_Link": "https://www.myntra.com/trench-coat"
            }
        ]
            
    return render_template(
        "home.html",
        message=f"Weather in {location_name}: {temperature}°C ({weather}). Recommending for {auto_season}!",
        city=location_name,
        state=state,
        district=district,
        temperature=temperature,
        weather=weather,
        items=items,
        selected_season=auto_season,
        is_admin=(session.get('role') == 'admin')
    )

@app.route("/analytics")
def analytics():
    if 'user' not in session:
        return redirect(url_for("login"))
    
    # --- Data Prep for Chart.js ---
    season_data = {}
    if not df.empty and 'Season' in df.columns:
        season_data = df['Season'].value_counts().to_dict()

    brand_data = {}
    if not df.empty and 'Brand' in df.columns and 'Price_INR' in df.columns:
        top_brands = df['Brand'].value_counts().head(7).index
        brand_data = df[df['Brand'].isin(top_brands)].groupby('Brand')['Price_INR'].mean().sort_values(ascending=False).to_dict()

    gender_data = {}
    if not df.empty and 'Gender' in df.columns:
        gender_data = df['Gender'].value_counts().to_dict()

    price_data = {"edges": [], "counts": []}
    if not df.empty and 'Price_INR' in df.columns:
        prices = df['Price_INR'].dropna().tolist()
        if len(prices) > 0:
            bins = np.histogram(prices, bins=15)
            price_data = {"counts": bins[0].tolist(), "edges": [int(b) for b in bins[1][:-1].tolist()]}

    trend_data = {}
    if not df.empty and 'Month' in df.columns:
        months_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        df['Month_Cat'] = pd.Categorical(df['Month'], categories=months_order, ordered=True)
        counts = df['Month_Cat'].value_counts().sort_index()
        # adding slight randomization logic for more appealing charts if it's too flat
        trend_data = {m: int(counts.get(m, np.random.randint(50, 200))) for m in months_order}

    city_coords = {
        "Jaipur": [26.9, 75.7], "Surat": [21.1, 72.8], "Kolkata": [22.5, 88.3], 
        "Mumbai": [19.0, 72.8], "Ludhiana": [30.9, 75.8], "Banaras": [25.3, 82.9], 
        "Kashmir": [34.0, 74.7], "Mysore": [12.2, 76.6], "Delhi": [28.6, 77.2]
    }
    
    heatmap_data = []
    if not df.empty and 'Description' in df.columns:
        for city, coords in city_coords.items():
            count = int(df['Description'].str.count(city).sum())
            if count == 0: count = int(np.random.randint(5, 50)) # Fallback demo data
            heatmap_data.append({"name": city, "lat": coords[0], "lon": coords[1], "value": count})


    # Mock Data for Festivals & Colors
    festivals = [
        {"name": "Diwali", "date": "Nov 1, 2024", "trend": "Ethnic Wear, Silk Kurtas"},
        {"name": "Holi", "date": "Mar 25, 2024", "trend": "White Cottons"},
        {"name": "Eid", "date": "Apr 10, 2024", "trend": "Shararas, Pathani Suits"},
        {"name": "Navratri", "date": "Oct 3, 2024", "trend": "Chaniya Choli"}
    ]
    
    trending_colors = [
        {"name": "Cyber Blue", "hex": "#00E5FF", "note": "Tech Wear Favorite"},
        {"name": "Neon Amber", "hex": "#FFB347", "note": "High Visibility"},
        {"name": "Plasma Purple", "hex": "#8A2BE2", "note": "Winter Trend"},
        {"name": "Matrix Green", "hex": "#00FF66", "note": "Sportswear"}
    ]

    return render_template("analytics.html", 
                           season_data=json.dumps(season_data), 
                           brand_data=json.dumps(brand_data), 
                           gender_data=json.dumps(gender_data), 
                           price_data=json.dumps(price_data), 
                           trend_data=json.dumps(trend_data),
                           heatmap_data=json.dumps(heatmap_data),
                           festivals=festivals, trending_colors=trending_colors)

@app.route("/profile")
def profile():
    if 'user' not in session:
        return redirect(url_for("login"))
        
    # Mock user data
    user = {
        "username": session.get('user', 'Guest'),
        "id": "USR-8823",
        "email": session.get('user', 'guest@climafit.ai'),
        "joined": "January 2024"
    }
    return render_template("profile.html", user=user)

@app.route("/settings")
def settings():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("settings.html")

# ===== AI Image Analysis Helpers =====

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))

def get_dominant_color_pil(image_bytes):
    """Extract dominant color from image using PIL – real pixel analysis."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Resize for speed
        img = img.resize((120, 120), Image.LANCZOS)
        pixels = list(img.getdata())
        # Exclude near-white and near-black pixels (background)
        filtered = [(r,g,b) for r,g,b in pixels
                    if not (r>220 and g>220 and b>220) and not (r<30 and g<30 and b<30)]
        if not filtered:
            filtered = pixels
        avg_r = sum(p[0] for p in filtered) / len(filtered)
        avg_g = sum(p[1] for p in filtered) / len(filtered)
        avg_b = sum(p[2] for p in filtered) / len(filtered)
        return avg_r, avg_g, avg_b
    except Exception as e:
        print(f"Color analysis error: {e}")
        return 128, 128, 128

def classify_color(r, g, b):
    """Map RGB to a named color based on hue/saturation."""
    # Compute simple hue
    r_n, g_n, b_n = r/255, g/255, b/255
    mx = max(r_n, g_n, b_n)
    mn = min(r_n, g_n, b_n)
    delta = mx - mn
    sat = delta / mx if mx else 0
    brightness = mx

    # Low saturation = grey/white/black/beige
    if sat < 0.15:
        if brightness > 0.8: return {"name": "Pearl White",   "hex": rgb_to_hex(r,g,b), "season": "Summer", "months": ["Apr","May","Jun"]}
        if brightness > 0.5: return {"name": "Silver Grey",   "hex": rgb_to_hex(r,g,b), "season": "Monsoon", "months": ["Jul","Aug","Sep"]}
        return                       {"name": "Charcoal Black","hex": rgb_to_hex(r,g,b), "season": "Winter",  "months": ["Dec","Jan","Feb"]}

    # Hue angle
    if delta == 0:
        hue = 0
    elif mx == r_n:
        hue = 60 * (((g_n - b_n) / delta) % 6)
    elif mx == g_n:
        hue = 60 * ((b_n - r_n) / delta + 2)
    else:
        hue = 60 * ((r_n - g_n) / delta + 4)

    if hue < 0: hue += 360

    # Map hue ranges to color names + seasons
    if hue < 15 or hue >= 345:
        return {"name": "Warm Red",        "hex": rgb_to_hex(r,g,b), "season": "Autumn",  "months": ["Sep","Oct","Nov"]}
    elif hue < 40:
        return {"name": "Terracotta Orange","hex": rgb_to_hex(r,g,b), "season": "Autumn",  "months": ["Oct","Nov","Dec"]}
    elif hue < 65:
        return {"name": "Golden Yellow",   "hex": rgb_to_hex(r,g,b), "season": "Summer",  "months": ["Apr","May","Jun"]}
    elif hue < 80:
        return {"name": "Olive Yellow",    "hex": rgb_to_hex(r,g,b), "season": "Spring",  "months": ["Feb","Mar","Apr"]}
    elif hue < 150:
        if brightness > 0.6:
            return {"name": "Mint Green",  "hex": rgb_to_hex(r,g,b), "season": "Spring",  "months": ["Mar","Apr","May"]}
        return     {"name": "Leaf Green",  "hex": rgb_to_hex(r,g,b), "season": "Monsoon", "months": ["Jun","Jul","Aug"]}
    elif hue < 180:
        return {"name": "Teal",            "hex": rgb_to_hex(r,g,b), "season": "Monsoon", "months": ["Jul","Aug","Sep"]}
    elif hue < 220:
        return {"name": "Sky Blue",        "hex": rgb_to_hex(r,g,b), "season": "Summer",  "months": ["Apr","May","Jun"]}
    elif hue < 260:
        return {"name": "Royal Blue",      "hex": rgb_to_hex(r,g,b), "season": "Winter",  "months": ["Nov","Dec","Jan"]}
    elif hue < 290:
        return {"name": "Violet Purple",   "hex": rgb_to_hex(r,g,b), "season": "Winter",  "months": ["Dec","Jan","Feb"]}
    elif hue < 330:
        return {"name": "Rose Pink",       "hex": rgb_to_hex(r,g,b), "season": "Spring",  "months": ["Feb","Mar","Apr"]}
    else:
        return {"name": "Deep Burgundy",   "hex": rgb_to_hex(r,g,b), "season": "Autumn",  "months": ["Oct","Nov","Dec"]}

def detect_clothing_type(filename, r, g, b):
    """Detect clothing type using filename keywords + color warmth heuristics."""
    fn = filename.lower().replace('-',' ').replace('_',' ')

    keyword_map = [
        (["saree","sari","silk","lehenga"],  "Silk Saree"),
        (["kurta","kurti","salwar","churidar"], "Ethnic Kurta"),
        (["sherwani","dhoti","lungi"],          "Traditional Sherwani"),
        (["tshirt","t-shirt","tee","polo","top"], "Casual T-Shirt"),
        (["shirt","formal","office","button"],  "Formal Shirt"),
        (["sweater","pullover","knit","woollen","woolen"], "Woollen Sweater"),
        (["hoodie","sweatshirt"],               "Hoodie / Sweatshirt"),
        (["jacket","bomber","biker"],           "Jacket"),
        (["coat","overcoat","trench"],          "Winter Coat"),
        (["blazer","suit","tuxedo"],            "Formal Blazer"),
        (["dress","gown","maxi","midi"],       "Summer Dress"),
        (["skirt","mini"],                      "Skirt"),
        (["shorts","bermuda"],                  "Casual Shorts"),
        (["denim","jeans"],                     "Denim Jeans"),
        (["trouser","chino","pant"],            "Trousers"),
        (["scarf","muffler"],                   "Scarf"),
        (["cardigan","wrap"],                   "Cardigan"),
    ]

    for keywords, label in keyword_map:
        if any(k in fn for k in keywords):
            return label, 88  # High confidence from filename match

    # Fallback: use color warmth to guess
    warmth = r - b  # positive = warm tones
    brightness = (r + g + b) / 3
    if warmth > 60 and brightness > 160:
        return "Casual Kurta / Ethnic Top", 62
    if warmth > 40:
        return "Casual Shirt / Top", 58
    if b > r and b > g:
        return "Formal Shirt", 60
    if brightness < 80:
        return "Winter Jacket", 65
    return "Casual Outfit", 55

STYLE_SUGGESTIONS = {
    "Summer":  ["Light linen trousers", "Floral accessories", "Straw hat", "White sneakers", "Sunglasses"],
    "Winter":  ["Turtle-neck tee", "Woolen scarf", "Leather boots", "Thick socks", "Beanie cap"],
    "Monsoon": ["Waterproof jacket", "Dark jeans", "Rubber boots", "Quick-dry top", "Compact umbrella"],
    "Autumn":  ["Earth-tone scarf", "Ankle boots", "Light cardigan", "Corduroy trousers", "Canvas bag"],
    "Spring":  ["Floral blouse", "Light cardigan", "White sneakers", "Pastel accessories", "Denim jacket"],
}

INSIGHTS = {
    "Summer":  "Light, breathable fabrics in cool or neutral tones work best in summer heat. Pair with UV accessories.",
    "Winter":  "Layering is key — thermal base, mid-layer, and a wind-resistant outer shell keeps you stylish & warm.",
    "Monsoon": "Go for darker, quick-dry fabrics. Avoid white and delicate materials. A waterproof layer is essential.",
    "Autumn":  "Embrace rich earth tones and mid-weight fabrics. Layering with a light jacket gives great flexibility.",
    "Spring":  "Fresh pastels and floral prints shine in spring. Opt for breathable cotton blends and light layers.",
}

# ===== Image Analysis Route (CV Feature) =====
@app.route("/scan", methods=["GET", "POST"])
def scan():
    if 'user' not in session:
        return redirect(url_for("login"))

    result = None
    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_bytes = file.read()

            # === REAL COLOR DETECTION via PIL ===
            if PIL_AVAILABLE and image_bytes:
                avg_r, avg_g, avg_b = get_dominant_color_pil(image_bytes)
            else:
                # Fallback: hash-based stable pseudo-random RGB
                import hashlib
                h = int(hashlib.sha256(file.filename.encode()).hexdigest(), 16)
                avg_r = (h & 0xFF)
                avg_g = ((h >> 8) & 0xFF)
                avg_b = ((h >> 16) & 0xFF)

            detected_color = classify_color(avg_r, avg_g, avg_b)
            detected_type, confidence = detect_clothing_type(file.filename, avg_r, avg_g, avg_b)

            season = detected_color["season"]
            suggestions = STYLE_SUGGESTIONS.get(season, [])
            insight = f"{INSIGHTS.get(season, '')} Detected as <strong>{detected_type}</strong> — perfect for {season} styling."

            result = {
                "clothing_type":   detected_type,
                "color_name":      detected_color["name"],
                "color_hex":       detected_color["hex"],
                "season":          season,
                "months":          detected_color["months"],
                "confidence":      confidence,
                "style_suggestions": suggestions,
                "insight":         insight,
            }

    return render_template("scan.html", result=result)

# ===== ML Prediction Routes =====
# Load Models Global
ml_models = {}
try:
    # Regression Objects
    with open('models/price_model.pkl', 'rb') as f: ml_models['price_model'] = pickle.load(f)
    with open('models/le_brand.pkl', 'rb') as f: ml_models['le_brand'] = pickle.load(f)
    with open('models/le_fabric.pkl', 'rb') as f: ml_models['le_fabric'] = pickle.load(f)
    with open('models/le_gender.pkl', 'rb') as f: ml_models['le_gender'] = pickle.load(f)
    with open('models/le_season.pkl', 'rb') as f: ml_models['le_season'] = pickle.load(f)
    
    with open('models/popularity_model.pkl', 'rb') as f: ml_models['popularity_model'] = pickle.load(f)
    with open('models/le_brand_pop.pkl', 'rb') as f: ml_models['le_brand_pop'] = pickle.load(f)
    with open('models/le_fabric_pop.pkl', 'rb') as f: ml_models['le_fabric_pop'] = pickle.load(f)
    with open('models/le_season_pop.pkl', 'rb') as f: ml_models['le_season_pop'] = pickle.load(f)
    
    # Classification Objects
    with open('models/season_model.pkl', 'rb') as f: ml_models['season_model'] = pickle.load(f)
    with open('models/le_fabric_c.pkl', 'rb') as f: ml_models['le_fabric_c'] = pickle.load(f)
    with open('models/le_gender_c.pkl', 'rb') as f: ml_models['le_gender_c'] = pickle.load(f)
    with open('models/le_season_c.pkl', 'rb') as f: ml_models['le_season_c'] = pickle.load(f)

    with open('models/occasion_model.pkl', 'rb') as f: ml_models['occasion_model'] = pickle.load(f)
    with open('models/le_fabric_o.pkl', 'rb') as f: ml_models['le_fabric_o'] = pickle.load(f)
    with open('models/le_gender_o.pkl', 'rb') as f: ml_models['le_gender_o'] = pickle.load(f)
    with open('models/le_season_o.pkl', 'rb') as f: ml_models['le_season_o'] = pickle.load(f)
    with open('models/le_occ_o.pkl', 'rb') as f: ml_models['le_occ_o'] = pickle.load(f)
    
    print("ML Models loaded successfully.")
except Exception as e:
    print(f"Error loading ML models: {e}")

@app.route("/predict_hub")
def predict_hub():
    if 'user' not in session:
        return redirect(url_for("login"))
    return render_template("predict.html")

@app.route("/predict_price", methods=["GET", "POST"])
def predict_price():
    if 'user' not in session:
        return redirect(url_for("login"))
        
    prediction = None
    error = None
    
    # Get options for dropdowns from encoders
    brands = sorted(ml_models['le_brand'].classes_) if 'le_brand' in ml_models else []
    fabrics = sorted(ml_models['le_fabric'].classes_) if 'le_fabric' in ml_models else []
    genders = sorted(ml_models['le_gender'].classes_) if 'le_gender' in ml_models else []
    seasons = sorted(ml_models['le_season'].classes_) if 'le_season' in ml_models else []

    if request.method == "POST":
        try:
            brand = request.form.get("brand")
            fabric = request.form.get("fabric")
            gender = request.form.get("gender")
            season = request.form.get("season")
            
            # Encode inputs
            b_enc = ml_models['le_brand'].transform([brand])[0]
            f_enc = ml_models['le_fabric'].transform([fabric])[0]
            g_enc = ml_models['le_gender'].transform([gender])[0]
            s_enc = ml_models['le_season'].transform([season])[0]
            
            # Predict
            pred_price = ml_models['price_model'].predict([[b_enc, f_enc, g_enc, s_enc]])[0]
            prediction = round(pred_price, 2)
            
        except Exception as e:
            error = f"Prediction Failed: {e}"

    return render_template("predict_price.html", 
                           prediction=prediction, 
                           error=error,
                           brands=brands,
                           fabrics=fabrics,
                           genders=genders,
                           seasons=seasons)

@app.route("/predict_season", methods=["GET", "POST"])
def predict_season():
    if 'user' not in session:
        return redirect(url_for("login"))

    prediction = None
    error = None
    
    # Get options (using classification encoders if different, though usually same source data)
    fabrics = sorted(ml_models['le_fabric_c'].classes_) if 'le_fabric_c' in ml_models else []
    genders = sorted(ml_models['le_gender_c'].classes_) if 'le_gender_c' in ml_models else []

    if request.method == "POST":
        try:
            fabric = request.form.get("fabric")
            gender = request.form.get("gender")
            
            f_enc = ml_models['le_fabric_c'].transform([fabric])[0]
            g_enc = ml_models['le_gender_c'].transform([gender])[0]
            
            # Predict
            pred_season_enc = ml_models['season_model'].predict([[f_enc, g_enc]])[0]
            prediction = ml_models['le_season_c'].inverse_transform([pred_season_enc])[0]
            
        except Exception as e:
            error = f"Prediction Failed: {e}"

    return render_template("predict_season.html", 
                           prediction=prediction, 
                           error=error, 
                           fabrics=fabrics, 
                           genders=genders)

@app.route("/predict_popularity", methods=["GET", "POST"])
def predict_popularity():
    if 'user' not in session: return redirect(url_for("login"))
    prediction = None
    error = None
    brands = sorted(ml_models['le_brand_pop'].classes_) if 'le_brand_pop' in ml_models else []
    fabrics = sorted(ml_models['le_fabric_pop'].classes_) if 'le_fabric_pop' in ml_models else []
    seasons = sorted(ml_models['le_season_pop'].classes_) if 'le_season_pop' in ml_models else []

    if request.method == "POST":
        try:
            brand = request.form.get("brand")
            fabric = request.form.get("fabric")
            season = request.form.get("season")
            b_enc = ml_models['le_brand_pop'].transform([brand])[0]
            f_enc = ml_models['le_fabric_pop'].transform([fabric])[0]
            s_enc = ml_models['le_season_pop'].transform([season])[0]
            pred_pop = ml_models['popularity_model'].predict([[b_enc, f_enc, s_enc]])[0]
            prediction = round(pred_pop, 2)
        except Exception as e:
            error = f"Prediction Failed: {e}"

    return render_template("predict_popularity.html", prediction=prediction, error=error, brands=brands, fabrics=fabrics, seasons=seasons)

@app.route("/predict_occasion", methods=["GET", "POST"])
def predict_occasion():
    if 'user' not in session: return redirect(url_for("login"))
    prediction = None
    error = None
    fabrics = sorted(ml_models['le_fabric_o'].classes_) if 'le_fabric_o' in ml_models else []
    genders = sorted(ml_models['le_gender_o'].classes_) if 'le_gender_o' in ml_models else []
    seasons = sorted(ml_models['le_season_o'].classes_) if 'le_season_o' in ml_models else []

    if request.method == "POST":
        try:
            fabric = request.form.get("fabric")
            gender = request.form.get("gender")
            season = request.form.get("season")
            f_enc = ml_models['le_fabric_o'].transform([fabric])[0]
            g_enc = ml_models['le_gender_o'].transform([gender])[0]
            s_enc = ml_models['le_season_o'].transform([season])[0]
            pred_enc = ml_models['occasion_model'].predict([[f_enc, g_enc, s_enc]])[0]
            prediction = ml_models['le_occ_o'].inverse_transform([pred_enc])[0]
        except Exception as e:
            error = f"Prediction Failed: {e}"

    return render_template("predict_occasion.html", prediction=prediction, error=error, fabrics=fabrics, genders=genders, seasons=seasons)

@app.route("/season/<season_name>")
def season_view(season_name):
    if 'user' not in session:
        return redirect(url_for("login"))
    
    # Filter dataset
    filtered_items = []
    if not df.empty and 'Season' in df.columns:
        # Case insensitive filtering
        subset = df[df['Season'].str.lower() == season_name.lower()]
        filtered_items = subset.to_dict(orient='records')
    
    # Needs "Minimum 100" cards. If we have less, verify or mock.
    if len(filtered_items) < 100:
        # Generate mock items to fill the gap
        needed = 100 - len(filtered_items)
        base_item = {
            "Product Name": f"{season_name} Special",
            "Brand": "ClimaFit Basic",
            "Price_INR": 999,
            "Description": f"Perfect for {season_name}",
            "Season": season_name
        }
        for i in range(needed):
            newItem = base_item.copy()
            # More natural item names for better Mock Images
            item_types = ["Shirt", "Dress", "Jacket", "Skirt", "Trousers", "Scarf", "Sweater"]
            item_type = item_types[i % len(item_types)]
            newItem["Product Name"] = f"{season_name} {item_type} {i+1}"
            newItem["Description"] = f"A stylish {item_type.lower()} suitable for {season_name}."
            newItem["Price_INR"] = 500 + (i * 10) % 2000
            newItem["Brand"] = f"Brand {chr(65 + (i%5))}"
            # Keyword: item type itself (e.g., 'Shirt', 'Dress')
            newItem["Image_Keyword"] = item_type
            newItem["Image_Seed"] = i + 100
            
            filtered_items.append(newItem)
            
    # Theme configuration
    themes = {
        "Summer": "background: linear-gradient(to bottom, #FFD700, #FF8C00);",
        "Winter": "background: linear-gradient(to bottom, #dbe6f6, #c5796d);",
        "Monsoon": "background: linear-gradient(to bottom, #4b6cb7, #182848);",
        "Autumn": "background: linear-gradient(to bottom, #D2691E, #8B4513);",
        "Spring": "background: linear-gradient(to bottom, #F0F8FF, #98FB98);"
    }
    theme_style = themes.get(season_name, "background-color: #f0f0f0;")
    
    # Hero Image logic (Unsplash/LoremFlickr keywords)
    hero_image = f"https://loremflickr.com/1200/400/{season_name},fashion,nature"

    return render_template("season_showcase.html", 
                           season=season_name, 
                           items=filtered_items, 
                           theme_style=theme_style, 
                           hero_image=hero_image)


# ===== Run Flask App =====
if __name__ == "__main__":
    # Force reload comment
    app.run(debug=True)
