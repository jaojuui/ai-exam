import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ========================
# 1. โหลดข้อมูล
# ========================
df = pd.read_csv("data_missing.csv", na_values=["--"])

# เติมค่า missing ใน y (Rain)
df["Rain"] = df["Rain"].fillna(df["Rain"].mean())

# ========================
# 2. แยก features / label
# ========================
X = df.drop("Rain", axis=1)
y = df["Rain"]

# แปลง string → ตัวเลข
X = pd.get_dummies(X, columns=["Zone"])

# ========================
# 3. แบ่ง train / test
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 4. จัดการ missing (เฉพาะ X)
# ========================
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ========================
# 5. scaling
# ========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================
# 6. train model (Regression)
# ========================
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ========================
# 7. evaluate
# ========================
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("Test size:", len(X_test))

# ========================
# 8. save model + tools
# ========================
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")

print("Saved all models!")

# ========================
# 9. โหลดกลับมาทดสอบ
# ========================
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

print(model)
print(model.get_params())