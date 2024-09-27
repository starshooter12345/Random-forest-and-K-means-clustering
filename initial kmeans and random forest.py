from google.colab import files

# Upload the Dataset.csv and Features_Details.pdf
uploaded = files.upload()

# Display the uploaded files
for filename in uploaded.keys():
    print(f'Uploaded file: {filename}')


import pandas as pd

# Load the dataset
df = pd.read_csv('Dataset.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Check basic information about the dataset
print(df.info())

# Load the dataset, forcing all columns to be interpreted as strings
df = pd.read_csv('Dataset.csv', dtype=str)

# Convert the relevant columns back to integers (except 'class' which seems to be a label)
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Check again for any issues
print(df.info())


# Convert all columns except 'class' to numeric (force non-numeric to NaN)
df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

# Check again for any issues
print(df.info())


# Check for missing values
missing_values = df.isnull().sum()

# Display columns with missing values
print(missing_values[missing_values > 0])

# Fill the missing value in the column 'TelephonyManager.getSimCountryIso' with 0
df['TelephonyManager.getSimCountryIso'].fillna(0, inplace=True)

# Check if there are any more missing values
print(df.isnull().sum().sum())  # Should output 0 if no missing values remain

# Convert the 'class' column to numerical values
df['class'] = df['class'].map({'S': 1, 'B': 0})

# Verify the conversion
print(df['class'].value_counts())

# Separate features (X) and labels (y)
X = df.iloc[:, :-1]  # All columns except the last one ('class' column)
y = df['class']      # The last column ('class')

# Check the shapes of X and y
print(X.shape)  # Should output (16300, 215)
print(y.shape)  # Should output (16300,)

from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

from sklearn.cluster import KMeans

# Train the K-Means model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)

# Predict clusters for the training and testing data
X_train_clusters = kmeans.predict(X_train)
X_test_clusters = kmeans.predict(X_test)


from sklearn.cluster import KMeans

# Train the K-Means model with 2 clusters and set n_init explicitly
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

# Predict clusters for the training and testing data
X_train_clusters = kmeans.predict(X_train)
X_test_clusters = kmeans.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train the Random Forest classifier using the clusters as features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_clusters.reshape(-1, 1), y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test_clusters.reshape(-1, 1))

# Evaluate the model
print(classification_report(y_test, y_pred))

import pickle

# Train the K-Means model with 2 clusters and set n_init explicitly
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train)

# Predict clusters for the training and testing data
X_train_clusters = kmeans.predict(X_train)
X_test_clusters = kmeans.predict(X_test)

# Save the K-Means model to a .pkl file
with open('kmeans_model.pkl', 'wb') as kmeans_file:
    pickle.dump(kmeans, kmeans_file)

# Train the Random Forest classifier using the clusters as features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_clusters.reshape(-1, 1), y_train)

# Save the Random Forest model to a .pkl file
with open('rf_model.pkl', 'wb') as rf_file:
    pickle.dump(rf, rf_file)

# Make predictions on the test set
y_pred = rf.predict(X_test_clusters.reshape(-1, 1))

# Evaluate the model
print(classification_report(y_test, y_pred))


# Loading the K-Means model
with open('kmeans_model.pkl', 'rb') as kmeans_file:
    kmeans_model = pickle.load(kmeans_file)

# Loading the Random Forest model
with open('rf_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

    !pip install fastapi
!pip install uvicorn
!pip install pyngrok
!pip install nest-asyncio
!pip install scikit-learn
!pip install pydantic
!pip install requests

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

app = FastAPI()

# CORS middleware setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Example: Loading trained models
# Replace 'model_path.pkl' with your actual model paths
with open('rf_model.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

with open('kmeans_model.pkl', 'rb') as kmeans_file:
    kmeans_model = pickle.load(kmeans_file)
@app.post("/analyze_apk")
async def analyze_apk(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    # Extract features from APK (You need to implement this function)
    # Replace with actual feature extraction code
    features = extract_features(file_location)  # Implement this function based on your APK analysis logic

    # Scale the features if needed
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform([features])

    # Predict cluster
    cluster = kmeans_model.predict(features_scaled)

    # Predict using Random Forest
    prediction = rf_model.predict(cluster.reshape(-1, 1))

    # Result interpretation
    result = "Malicious" if prediction[0] == 1 else "Benign"

    # Additional analysis (replace with actual analysis details)
    malware_percentage = 0.8  # Dummy value, replace with actual analysis output
    malware_types = ["Type A", "Type B"]  # Replace with actual types from analysis

    return {
        "result": result,
        "malware_percentage": malware_percentage,
        "malware_types": malware_types
    }

# Install pyngrok
!pip install pyngrok

# Install the ngrok executable
!wget -q -O ngrok.zip https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip -o ngrok.zip

# Replace 'YOUR_NGROK_AUTH_TOKEN' with your actual Ngrok authtoken from the dashboard
!./ngrok config add-authtoken ""

from pyngrok import ngrok
import nest_asyncio

# Start the ngrok tunnel on port 8000
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)

# Allow nest_asyncio to run the event loop in Colab
nest_asyncio.apply()


Public URL: https://59e3-34-168-247-61.ngrok-free.app

import uvicorn
from fastapi import FastAPI

app = FastAPI()

# Example FastAPI route
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Run Uvicorn server on port 8000
uvicorn.run(app, host="0.0.0.0", port=8000)


import requests

# Replace <your-ngrok-url> with the actual URL provided by ngrok
url = "https://https://59e3-34-168-247-61.ngrok-free.app//analyze_apk"

# Upload an APK file for testing
files = {'file': open('path_to_your_apk.apk', 'rb')}  # Replace with the path to your APK file

response = requests.post(url, files=files)
print(response.json())

pip install python-multipart
