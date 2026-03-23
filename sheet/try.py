import joblib
import numpy as np

# โหลด
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ข้อมูลใหม่
new_data = np.array([[5.7, 2.3, 3.4, 1]])

# scale
new_data = scaler.transform(new_data)

# predict
prediction = model.predict(new_data)

print("Prediction:", prediction)