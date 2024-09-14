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