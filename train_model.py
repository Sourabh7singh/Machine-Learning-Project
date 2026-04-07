import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("🔄 Loading dataset (this might take a minute since it's 700MB)...")
data = pd.read_csv('Accident_Information.csv', low_memory=False)

print("🧹 Cleaning data and dropping columns...")
data.drop('Accident_Index', axis=1, inplace=True)
data.replace("Unclassified", np.nan, inplace=True)
data.replace("Data missing or out of range", np.nan, inplace=True)

columns_to_drop = [
    '1st_Road_Class', '2nd_Road_Class', 'Carriageway_Hazards', 
    'Junction_Control', 'LSOA_of_Accident_Location', 
    'Special_Conditions_at_Site', 'Location_Easting_OSGR', 
    'Location_Northing_OSGR'
]
data.drop(columns=columns_to_drop, inplace=True)

print("🕒 Formatting Time...")
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M', errors='coerce')
data['Time_of_Day'] = data['Time'].dt.hour.apply(
    lambda x: 'Morning' if pd.notnull(x) and 5 <= x < 12 else ('Evening' if pd.notnull(x) and 12 <= x < 18 else 'Night')
)
data.drop(columns=['Time', 'Date'], inplace=True)

print("🔠 Encoding Categorical columns...")
categorical_columns = [
    'Accident_Severity', 'Day_of_Week', 'Junction_Detail', 
    'Light_Conditions', 'Local_Authority_(District)', 
    'Local_Authority_(Highway)', 'Police_Force', 
    'Road_Surface_Conditions', 'Road_Type', 
    'Urban_or_Rural_Area', 'Weather_Conditions', 
    'InScotland', 'Year', 'Time_of_Day'
]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    # converting to string helps avoid errors when NaN is mixed with strings
    data[col] = data[col].astype(str) 
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("📉 Filling missing values with the Mode...")
for column in data.columns:
    mode_value = data[column].mode().iloc[0]
    data[column].fillna(mode_value, inplace=True)

target_column = 'Accident_Severity'
X = data.drop(columns=[target_column])
y = data[target_column]

print("🚀 Training Random Forest Model... (Might take a few minutes)")
# Setting n_jobs=-1 to use all CPU cores and speed up training significantly
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X, y)

print("💾 Saving Model & Encoders to disk...")
os.makedirs('artifacts', exist_ok=True)
joblib.dump(clf, 'artifacts/rf_model.pkl')
joblib.dump(label_encoders, 'artifacts/label_encoders.pkl')

# Save the feature columns so the web app knows what order to feed them in
feature_columns = list(X.columns)
joblib.dump(feature_columns, 'artifacts/feature_columns.pkl')

print("✅ Complete! You can now start the FastAPI server.")
