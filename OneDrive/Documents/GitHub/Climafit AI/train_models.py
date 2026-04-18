import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

csv_path = r"C:\Users\Khushi Wankhede\OneDrive\Desktop\climafit college major\Indian_Fashion_Dataset.csv"
try:
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Synthesize new targets if missing
if 'Popularity' not in df.columns:
    df['Popularity'] = np.random.randint(40, 100, size=len(df))
if 'Occasion' not in df.columns:
    occasions = ['Casual', 'Formal', 'Ethnic', 'Party', 'Sportswear']
    df['Occasion'] = np.random.choice(occasions, size=len(df))

# ==========================================
# 1. Regression Model: Price Prediction
# ==========================================
print("\n--- Training Price Prediction Model ---")
price_features = ['Brand', 'Fabric', 'Gender', 'Season']
df_price = df[price_features + ['Price_INR']].dropna()

le_brand = LabelEncoder()
le_fabric = LabelEncoder()
le_gender = LabelEncoder()
le_season = LabelEncoder()

df_price['Brand'] = le_brand.fit_transform(df_price['Brand'])
df_price['Fabric'] = le_fabric.fit_transform(df_price['Fabric'])
df_price['Gender'] = le_gender.fit_transform(df_price['Gender'])
df_price['Season'] = le_season.fit_transform(df_price['Season'])

X_p = df_price[price_features]
y_p = df_price['Price_INR']
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=20, random_state=42)
regressor.fit(X_train_p, y_train_p)
print(f"Price Prediction MAE: {mean_absolute_error(y_test_p, regressor.predict(X_test_p)):.2f}")

with open('models/price_model.pkl', 'wb') as f: pickle.dump(regressor, f)
with open('models/le_brand.pkl', 'wb') as f: pickle.dump(le_brand, f)
with open('models/le_fabric.pkl', 'wb') as f: pickle.dump(le_fabric, f)
with open('models/le_gender.pkl', 'wb') as f: pickle.dump(le_gender, f)
with open('models/le_season.pkl', 'wb') as f: pickle.dump(le_season, f)

# ==========================================
# 2. Classification Model: Season Recommendation
# ==========================================
print("\n--- Training Season Prediction Model ---")
season_features = ['Fabric', 'Gender'] 
df_season = df[season_features + ['Season']].dropna()

le_fabric_c = LabelEncoder()
le_gender_c = LabelEncoder()
le_season_c = LabelEncoder()

df_season['Fabric'] = le_fabric_c.fit_transform(df_season['Fabric'])
df_season['Gender'] = le_gender_c.fit_transform(df_season['Gender'])
df_season['Season'] = le_season_c.fit_transform(df_season['Season'])

X_s = df_season[season_features]
y_s = df_season['Season']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, y_s, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=20, random_state=42)
classifier.fit(X_train_s, y_train_s)
print(f"Season Prediction Accuracy: {accuracy_score(y_test_s, classifier.predict(X_test_s)):.2f}")

with open('models/season_model.pkl', 'wb') as f: pickle.dump(classifier, f)
with open('models/le_fabric_c.pkl', 'wb') as f: pickle.dump(le_fabric_c, f)
with open('models/le_gender_c.pkl', 'wb') as f: pickle.dump(le_gender_c, f)
with open('models/le_season_c.pkl', 'wb') as f: pickle.dump(le_season_c, f)

# ==========================================
# 3. Regression Model: Popularity Score Prediction
# ==========================================
print("\n--- Training Popularity Score Model ---")
pop_features = ['Brand', 'Fabric', 'Season']
df_pop = df[pop_features + ['Popularity']].dropna()

le_brand_pop = LabelEncoder()
le_fabric_pop = LabelEncoder()
le_season_pop = LabelEncoder()

df_pop['Brand'] = le_brand_pop.fit_transform(df_pop['Brand'])
df_pop['Fabric'] = le_fabric_pop.fit_transform(df_pop['Fabric'])
df_pop['Season'] = le_season_pop.fit_transform(df_pop['Season'])

X_pop = df_pop[pop_features]
y_pop = df_pop['Popularity']
X_train_pop, X_test_pop, y_train_pop, y_test_pop = train_test_split(X_pop, y_pop, test_size=0.2, random_state=42)

pop_reg = RandomForestRegressor(n_estimators=20, random_state=42)
pop_reg.fit(X_train_pop, y_train_pop)
print(f"Popularity Prediction MAE: {mean_absolute_error(y_test_pop, pop_reg.predict(X_test_pop)):.2f}")

with open('models/popularity_model.pkl', 'wb') as f: pickle.dump(pop_reg, f)
with open('models/le_brand_pop.pkl', 'wb') as f: pickle.dump(le_brand_pop, f)
with open('models/le_fabric_pop.pkl', 'wb') as f: pickle.dump(le_fabric_pop, f)
with open('models/le_season_pop.pkl', 'wb') as f: pickle.dump(le_season_pop, f)

# ==========================================
# 4. Classification Model: Occasion Classifier
# ==========================================
print("\n--- Training Occasion Classification Model ---")
occ_features = ['Fabric', 'Gender', 'Season']
df_occ = df[occ_features + ['Occasion']].dropna()

le_fabric_o = LabelEncoder()
le_gender_o = LabelEncoder()
le_season_o = LabelEncoder()
le_occ_o = LabelEncoder()

df_occ['Fabric'] = le_fabric_o.fit_transform(df_occ['Fabric'])
df_occ['Gender'] = le_gender_o.fit_transform(df_occ['Gender'])
df_occ['Season'] = le_season_o.fit_transform(df_occ['Season'])
df_occ['Occasion'] = le_occ_o.fit_transform(df_occ['Occasion'])

X_o = df_occ[occ_features]
y_o = df_occ['Occasion']
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_o, y_o, test_size=0.2, random_state=42)

occ_clf = RandomForestClassifier(n_estimators=20, random_state=42)
occ_clf.fit(X_train_o, y_train_o)
print(f"Occasion Classification Accuracy: {accuracy_score(y_test_o, occ_clf.predict(X_test_o)):.2f}")

with open('models/occasion_model.pkl', 'wb') as f: pickle.dump(occ_clf, f)
with open('models/le_fabric_o.pkl', 'wb') as f: pickle.dump(le_fabric_o, f)
with open('models/le_gender_o.pkl', 'wb') as f: pickle.dump(le_gender_o, f)
with open('models/le_season_o.pkl', 'wb') as f: pickle.dump(le_season_o, f)
with open('models/le_occ_o.pkl', 'wb') as f: pickle.dump(le_occ_o, f)

print("All advanced ML models generated successfully.")
