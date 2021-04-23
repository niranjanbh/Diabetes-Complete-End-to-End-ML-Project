import pandas as pd

data = pd.read_csv('C:/Users/NIRANJAN/Downloads/Intern Jun 2019/Datasets/14_Diabetes/Diabetes.csv')

data.columns = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Class')

data.loc[data.Class == 'YES', 'Class'] = 1
data.loc[data.Class == 'NO', 'Class'] = 0

# Elimination with Z Score
SD = data.std()
MN = data.mean()

data[((data > MN+3*SD)|(data < MN-3*SD)).any(axis=1)]

data = data[~((data > MN+3*SD)|(data < MN-3*SD)).any(axis=1)]
print(data.shape)


# Elimination with Interquartile Range
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data[((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
print(data.shape)