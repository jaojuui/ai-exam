import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ========================
# 1. โหลดข้อมูล
# ========================
df = pd.read_csv("iris.csv")

# แยก features / label
X = df.drop("species", axis=1)
y = df["species"]

# ========================
# 2. แบ่ง train / test
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 3. จัดการ missing data
# ========================
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ========================
# 4. scaling
# ========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========================
# 5. train model
# ========================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# ========================
# 6. evaluate
# ========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Test size:", len(X_test))

# ========================
# 7. save model + tools
# ========================
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")

print("Saved all models!")

# ========================
# 8. โหลดกลับมาทดสอบ
# ========================
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

print(model)
print(model.get_params())