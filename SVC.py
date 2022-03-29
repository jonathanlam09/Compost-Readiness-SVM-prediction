from calendar import c
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\Code projects\compost.csv")
x = df[['Temperature','Moisture_Level','Composted_Time']].values
y = df[['State_Of_Completion']].values
sc = StandardScaler()
X = sc.fit_transform(x)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = SVC(kernel="rbf", C=1, gamma=2)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

pd.to_pickle(model,r'D:\Code projects\updated_model.pickle')
model = pd.read_pickle(r'D:\Code projects\updated_model.pickle') 

Temperature = float(input("Enter Temperature\n"))
Moisture_Level = float(input("Enter Moisture percentage\n"))
Composted_Time = float(input("Enter composted time\n"))
result = model.predict(sc.transform([[Temperature, Moisture_Level, Composted_Time]]))

print("Temp: "+ str(Temperature))
print("Moist: "+ str(Moisture_Level))
print("Time: "+ str(Composted_Time))
print(result)
print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
print(metrics.classification_report(Y_test, Y_pred))
print(model.score(X_test, Y_test))
print(metrics.confusion_matrix(Y_test, Y_pred))