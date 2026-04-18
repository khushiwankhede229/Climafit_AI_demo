# ClimaFit AI - Project Documentation

## 1. Project Overview
ClimaFit AI is a smart fashion assistant that recommends clothing based on:
- **Live Weather**: Uses real-time weather data (Open-Meteo API) to suggest seasonal outfits.
- **Location**: Adapts to specific Indian cities and districts.
- **Fashion Trends**: Analyzes current market trends (simulated).
- **Computer Vision**: Scans uploaded images to detect clothing type and color suitability.

---

## 2. Project Structure
Here is a complete breakdown of every file in the project so you can study it.

### **Root Directory**
- **`app.py`**: The **Brain** of the project. This is a Flask (Python) web server.
  - **Routes**:
    - `/`: Login Page.
    - `/home`: Main Dashboard (Weather, Search, Recommendations).
    - `/scan`: Mock Computer Vision tool.
    - `/analytics`: Charts and Graphs.
    - `/settings`: Theme toggle.
  - **Key Functions**:
    - `get_weather(lat, lon)`: Fetches temperature.
    - `get_coordinates(district, state)`: Finds location.
    - `scan()`: The logic that "looks" at your uploaded image.

- **`generate_dataset.py`**: A script to create the `Indian_Fashion_Dataset.csv`. It generates 60,000 rows of realistic fashion data for 2024-2025.

- **`train_models.py`**: (Optional) Script to train Machine Learning models using the CSV data.

### **Templates (HTML Files)**
These are the frontend pages.
- **`home.html`**: The main landing page. Contains the weather widget and seasonal cards.
- **`analytics.html`**: The dashboard with charts (Charts are generated in Python using Matplotlib and sent as images).
- **`scan.html`**: The page where users upload images.
- **`settings.html`**: Page to toggle Light/Dark mode.
- **`login.html` / `signup.html`**: Authentication pages.

### **Static (CSS & Assets)**
- **`style.css`**: The main styling file. It uses **CSS Variables** (`--bg-primary`, `--text-primary`) to handle the Light/Dark mode switching.
- **`animations.css`**: Contains `@keyframes` for all the smooth fade-ins and mismatched effects.

---

## 3. How It Works (The Logic)

### **A. Image Handling (How are images added?)**
You asked: *"How are you adding images?"*
We use a **Dynamic Image Generation** technique. instead of storing thousands of images, we use a service called **LoremFlickr**.

** The Logic:**
1. In `app.py`, when we create a list of items (e.g., a "Summer Shirt"), we assign it a **Keyword**.
   - Example Code: `item["Image_Keyword"] = "Summer Shirt"`
2. In `home.html`, we create an image tag like this:
   ```html
   <img src="https://loremflickr.com/480/360/{{ item['Image_Keyword'] }}?lock={{ item['Image_Seed'] }}">
   ```
3. **What happens?** The browser asks LoremFlickr for an image matching "Summer Shirt". The `?lock=123` part ensures that the *same* product always shows the *same* image, so it doesn't change every refresh.

### **B. Center Alignment**
We used **CSS Grid** to center everything.
- `margin: 0 auto;`: This centers the main container box.
- `display: grid;`: This creates a grid layout.
- `justify-content: center;`: This forces all grid items (cards) to the middle of the screen.

### **C. Computer Vision (The Scan Feature)**
Since we cannot run a heavy AI model (like TensorFlow) easily in a lightweight demo, we created a **Smart Rule-Based System** in `app.py`.
1. It looks at the **filename** of the uploaded image (e.g., `red_saree.jpg`).
2. It detects keywords: "Saree" -> implies "Ethnic Wear" -> implies "Autumn/Festive".
3. It detects "Red" -> implies "Color Compatibility".
4. It returns a "Prediction" based on these rules. This simulates how a real AI would work.

---

## 4. How to Study This Project
1. **Start with `app.py`**: Read the `@app.route("/")` functions. Follow the path from Login -> Home.
2. **Check `style.css`**: Look for the `:root` section to understand how colors are defined.
3. **Experiment**: creating a new route or changing detailed keywords in `app.py`.

## 5. Deployment / Running
To run this project:
1. Open Terminal.
2. Type `python app.py`.
3. Open the link `http://127.0.0.1:5000` in your browser.

---
**ClimaFit AI - 2025**
