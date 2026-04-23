import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("Fertilizer_Prediction.csv")

# Encode categorical values
data['Soil Type'] = data['Soil Type'].astype('category').cat.codes
data['Crop Type'] = data['Crop Type'].astype('category').cat.codes

# Features & target
X = data.drop("Fertilizer Name", axis=1)
y = data["Fertilizer Name"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("fertilizer_model.pkl", "wb"))

print("✅ Fertilizer Recommendation Model Trained & Saved")
