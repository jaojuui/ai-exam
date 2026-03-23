import csv
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
df = pd.read_csv("iris.csv")




X = df.drop("species", axis=1)  # features
y = df["species"]               # label

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# model = LinearRegression()
# model.fit(X_train, y_train)

# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train, y_train)

# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(len(X_test))

# joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")
print(model)
print(model.get_params())
print(model.estimators_)