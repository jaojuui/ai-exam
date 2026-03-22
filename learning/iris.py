import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import os  # ใช้จัดการไฟล์และ path

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=2)
model = GaussianNB()

# Read data in from file
# sepal_length,sepal_width,petal_length,petal_width,species
os.chdir("./data")  # เปลี่ยนไปยังโฟลเดอร์ data
fname = 'iris.csv'  # กำหนดชื่อไฟล์
with open(fname) as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "detail": [float(cell) for cell in row[:4]],
            "label": row[4]
        })

# Separate data into training and testing groups
input_data = [row["detail"] for row in data]
output_data = [row["label"] for row in data]

X_training, X_testing, y_training, y_testing = train_test_split(
    input_data, output_data, test_size=0.4
)

# Fit model
model.fit(X_training, y_training)

# Make predictions on the testing set
predictions = model.predict(X_testing)

# Compute how well we performed
correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)

# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")