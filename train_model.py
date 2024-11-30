# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Synthetic weather data
data = {
    "temperature": [30, 35, 40, 45, 20, 25, 50, 15, 10, 5],
    "humidity": [70, 80, 85, 90, 60, 50, 95, 40, 30, 20],
    "wind_speed": [10, 15, 20, 25, 5, 7, 30, 2, 1, 0],
    "is_disaster": [0, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # 1 = disaster, 0 = no disaster
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[["temperature", "humidity", "wind_speed"]]
y = df["is_disaster"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
with open("models/disaster_model.pkl", "wb") as f:
    pickle.dump(model, f)
