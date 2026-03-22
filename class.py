import seaborn as sns  # ใช้สร้างกราฟ เช่น heatmap
import matplotlib.pyplot as plt  # ใช้วาดกราฟ
import os  # ใช้จัดการไฟล์และ path
import pandas as pd  # ใช้จัดการข้อมูลแบบตาราง
from sklearn.preprocessing import StandardScaler  # ใช้ปรับ scale ข้อมูล
import numpy as np  # ใช้คำนวณตัวเลข
from sklearn.linear_model import LinearRegression  # โมเดล Linear Regression

os.chdir("./data")  # เปลี่ยนไปยังโฟลเดอร์ data
fname = 'weather.csv'  # กำหนดชื่อไฟล์

if os.path.exists(fname):  # เช็คว่าไฟล์มีอยู่ไหม
    df = pd.read_csv(fname)  # อ่านไฟล์ CSV มาเป็น DataFrame
else:
    print('File does not exist.')  # ถ้าไม่มีไฟล์

print(df.head())  # แสดง 5 แถวแรกของข้อมูล

plt.figure(figsize=(15,15))  # กำหนดขนาดกราฟ
hmap = sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", annot_kws={"fontsize":16})
# สร้าง heatmap จาก correlation ของข้อมูล

hmap.set_xticklabels(hmap.get_xticklabels(), fontsize=16)  # ปรับขนาดตัวอักษรแกน X
hmap.set_yticklabels(hmap.get_yticklabels(), fontsize=16)  # ปรับขนาดตัวอักษรแกน Y

df1 = df.query("st==3001")  # เลือกข้อมูลที่ st = 3001
df1 = df1.drop(columns=['st','press'])  # ลบคอลัมน์ st และ press
df1.reset_index(drop=True, inplace=True)  # รีเซ็ต index ใหม่
df1.set_index(['year','month'], inplace=True)  # ตั้ง year, month เป็น index
print(df1.head(15))  # แสดง 15 แถวแรก

df2 = df.groupby(['st','year']).mean()  # จัดกลุ่มตาม st และ year แล้วหา mean
df2_drop = df2.drop(columns=['month'])  # ลบ column month
print(df2_drop.head(10))  # แสดง 10 แถวแรก

df3 = df.query("st==3003").drop(columns=['st'])  # เลือก st=3003 และลบ st
df4 = df3.groupby(['year']).mean()  # group ตาม year แล้วหา mean
df4_drop = df4.drop(columns=['month'])  # ลบ month
print(df4_drop.head(15))  # แสดง 15 แถวแรก

X = df4_drop[['rain','press','RH','wind']]  # ตัวแปร input (features)
print(X.head(5))  # แสดง 5 แถวแรกของ X

y = df4_drop.pop('temp')  # ตัวแปรเป้าหมาย (label)
print(y.head())  # แสดงค่า y

std_scaler = StandardScaler()  # สร้างตัว scaler

data_std = pd.DataFrame()  # สร้าง DataFrame ว่าง
data_std[df2_drop.columns] = pd.DataFrame(std_scaler.fit_transform(df2_drop))
# ปรับข้อมูล df2_drop ให้เป็น standard scale

print(data_std.info())  # ดูโครงสร้างข้อมูล
print(data_std.head(3))  # แสดง 3 แถวแรก

model = LinearRegression()  # สร้างโมเดล
model.fit(X, y)  # train โมเดลด้วย X และ y
print(model)  # แสดงโมเดล

model.coef_, model.intercept_  # ค่าสัมประสิทธิ์และค่า intercept
model.score(X, y)  # ค่า accuracy (R^2)

print(y)  # แสดงค่า y

y_predict = model.predict(X)  # ทำนายค่า y
print("="*40)  # พิมพ์เส้นคั่น
print('year\ty_true\t\ty_predict')  # header
print("="*40)

for i, y_true in enumerate(y):  # loop ค่า y จริง
    print(f'{i+2012}\t{y_true:.4f}\t\t{y_predict[i]:.4f}')
    # แสดงปี + ค่าจริง + ค่าทำนาย

plt.title('Heatmap Example', fontsize=16)  # ตั้งชื่อกราฟ
plt.xlabel('X-axis Label', fontsize=16)  # ชื่อแกน X
plt.ylabel('Y-axis Label', fontsize=16)  # ชื่อแกน Y
plt.show()  # แสดงกราฟ


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(X, y)




from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)



from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X, y)
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X, y)





from sklearn.svm import SVR

model = SVR()
model.fit(X, y)





from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor()
model.fit(X, y)