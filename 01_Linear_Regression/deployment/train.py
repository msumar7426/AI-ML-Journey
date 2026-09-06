"""
Rebuilds the exact pipeline from used_car_price_prediction.ipynb
(Stage 2 cleaning, Stage 4 feature engineering, Stage 5 preprocessing,
Stage 6 sklearn LinearRegression) and exports everything the frontend
needs to make a prediction: the trained model, the fitted scaler, the
exact training column layout, and the manufacturer/variant grouping
lists, so the app doesn't have to re-derive any of it.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "project", "used_car_listings_13_12_2025.csv"
)

df = pd.read_csv(CSV_PATH)

# ---- Stage 2: Data Cleaning ----
df_clean = df.drop_duplicates(keep="first")

df_clean["mileage"] = df_clean["mileage"].str.replace(",", "").str.replace(" km", "").astype(int)

df_clean["engine"] = df_clean["engine"].str.replace("cc", "")
df_clean.loc[df_clean["engine"] == "", "engine"] = "0"
df_clean["engine"] = df_clean["engine"].astype(int)

df_final = df_clean.drop_duplicates(
    subset=["manufacturer", "variant", "year", "engine", "fuelType", "transmission", "listingPrice"],
    keep="last",
)

df_final = df_final.drop(columns=["age"])

# Mileage placeholder + implausible values
df_final = df_final[df_final["mileage"] < 999000]
reference_year = 2025
age_years = (reference_year - df_final["year"]).clip(lower=1)
implied_km_per_year = df_final["mileage"] / age_years
df_final = df_final[implied_km_per_year <= 50000]

# Engine placeholder / implausible values
df_final = df_final[df_final["engine"] <= 6500]

# fuelType casing
df_final["fuelType"] = df_final["fuelType"].replace("Lpg", "LPG")

# name is redundant, dropped
df_final = df_final.drop(columns=["name"])

# ---- Stage 4: Feature Engineering ----
df_final["log_listingPrice"] = np.log(df_final["listingPrice"])

manufacturer_counts = df_final["manufacturer"].value_counts()
keep_manufacturers = manufacturer_counts[manufacturer_counts >= 100].index.tolist()
df_final["manufacturer_grouped"] = df_final["manufacturer"].where(
    df_final["manufacturer"].isin(keep_manufacturers), "Other"
)

variant_counts = df_final["variant"].value_counts()
keep_variants = variant_counts[variant_counts >= 20].index.tolist()
df_final["variant_grouped"] = df_final["variant"].where(
    df_final["variant"].isin(keep_variants), "Other"
)

# ---- Stage 5: Preprocessing ----
feature_cols = ["year", "mileage", "engine", "fuelType", "transmission", "manufacturer_grouped", "variant_grouped"]
X = df_final[feature_cols].copy()
y = df_final["log_listingPrice"].copy()

X = pd.get_dummies(X, columns=["fuelType", "transmission", "manufacturer_grouped", "variant_grouped"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = ["year", "mileage", "engine"]
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---- Stage 6: Modeling ----
model = LinearRegression()
model.fit(X_train, y_train)

r2 = model.score(X_test, y_test)
print(f"Sanity check R2 on held out test set: {r2:.4f}")

# ---- Export everything the app needs ----
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

# Dropdown options for the UI: real category values, sorted, "Other" excluded
# from the dropdown itself since a user picks a real make/model, grouping
# into Other happens automatically for anything the model didn't see enough of
manufacturer_options = sorted(df_final["manufacturer"].unique().tolist())
variant_options = sorted(df_final["variant"].unique().tolist())
fueltype_options = sorted(df_final["fuelType"].unique().tolist())
transmission_options = sorted(df_final["transmission"].unique().tolist())

# Manufacturer -> its own variants, so the UI can narrow the variant
# dropdown down once a manufacturer is picked instead of showing all 349
manufacturer_to_variants = (
    df_final.groupby("manufacturer")["variant"]
    .apply(lambda s: sorted(s.unique().tolist()))
    .to_dict()
)

# Variant -> the actual range of engine sizes seen for it in real listings.
# A Sportage isn't one fixed engine size across every year and trim, so this
# isn't used to hard-block anything, just to warn when someone types in a
# number nothing in the data ever had for that specific variant.
variant_engine_range = (
    df_final.groupby("variant")["engine"]
    .agg(["min", "max", "median"])
    .astype(int)
    .apply(lambda row: {"min": int(row["min"]), "max": int(row["max"]), "median": int(row["median"])}, axis=1)
    .to_dict()
)

meta = {
    "feature_columns": X.columns.tolist(),
    "numeric_cols": numeric_cols,
    "keep_manufacturers": keep_manufacturers,
    "keep_variants": keep_variants,
    "manufacturer_options": manufacturer_options,
    "variant_options": variant_options,
    "fueltype_options": fueltype_options,
    "transmission_options": transmission_options,
    "manufacturer_to_variants": manufacturer_to_variants,
    "variant_engine_range": variant_engine_range,
    "year_min": int(df_final["year"].min()),
    "year_max": int(df_final["year"].max()),
    "mileage_median": int(df_final["mileage"].median()),
    "engine_median": int(df_final["engine"].median()),
    "test_r2": round(float(r2), 4),
    "n_rows": int(len(df_final)),
}

with open("meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Exported model.joblib, scaler.joblib, meta.json")
print("manufacturers:", len(manufacturer_options), "variants:", len(variant_options))
