import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# โหลดข้อมูล
df = pd.read_csv("advertising.csv")

# X = df.select_dtypes(include=['int64', 'float64'])

# scale (สำคัญ)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# สร้าง model
model = KMeans(n_clusters=3, random_state=42)

# train (ไม่มี y)
model.fit(X_scaled)

# label ของแต่ละข้อมูล
labels = model.labels_

print(labels[:10])  # ดู 10 ตัวแรก

df["cluster"] = labels
print(df.head())
print(df["cluster"].value_counts())
print(df.groupby("cluster").mean())