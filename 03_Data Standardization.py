from pandas import read_csv
import pandas as pd

data = read_csv('C:/Users/NIRANJAN/Downloads/Intern Jun 2019/Datasets/14_Diabetes/Diabetes.csv')

data.columns = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Class')

data.loc[data.Class == 'YES', 'Class'] = 1
data.loc[data.Class == 'NO', 'Class'] = 0

X = data.iloc[:, [0,1,2,3,4,5,6,7]]
y = data.iloc[:, 8]

#Normalization
from sklearn.preprocessing import MinMaxScaler
X_scaler = MinMaxScaler().fit_transform(X)

#standardization
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler().fit_transform(X)