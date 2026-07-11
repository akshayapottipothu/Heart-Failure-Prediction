import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("dataset/heart.csv")

# Convert categorical columns into numbers
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the model
joblib.dump(model, "models/heart_model.pkl")

# Save the column names (needed later for prediction)
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

print("Model saved successfully!")