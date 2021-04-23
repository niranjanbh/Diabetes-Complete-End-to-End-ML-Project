import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

data = pd.read_csv('C:/Users/NIRANJAN/Downloads/Diabetes.csv')

print(data.columns) 

data.columns = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Class')

print(data.columns) 

print(data.shape)

print(data.head(10))

print(data.tail(2))

print(data.dtypes)
 
print(data.describe())

data.Pregnancies.max()

data['Pregnancies'].min()

data.BMI.mean()

data['DiabetesPedigreeFunction'].median()

data['DiabetesPedigreeFunction'].median()

data.sort_values('Glucose',ascending = False).head(20)

data.sort_values('BMI', ascending = True).head(20)

data.sort_values('BMI', ascending = False).tail(20)

data.loc[data.Insulin == 0]

data['BloodPressure'].loc[data.BloodPressure > 100]

data['BloodPressure'].loc[data.BloodPressure > 100].count()

data.loc[data.BloodPressure == 76]

data.loc[data.BloodPressure > 100, ['SkinThickness','BloodPressure' ]]

data.loc[(data.BMI > 29) & (data.BMI  < 30), ['BMI', 'SkinThickness']]

data['Glucose'].mean()

data['Glucose'].std()

data['Glucose'].mean() + data['Glucose'].std()

data['Glucose'].mean() - data['Glucose'].std()

data.loc[(data.Glucose < 152.86714944513622) & (data.Glucose > 88.92191305486378)]

data.loc[(data.Glucose < (data['Glucose'].mean() + data['Glucose'].std())) & (data.Glucose >(data['Glucose'].mean() - data['Glucose'].std()))]

(data< (data.mean() + 3 * data.std())) & (data >(data.mean() - 3 *data.std()))

data.loc[()]


data.isnull()
data.isnull().any()
data.isnull().sum()
data.isnull().sum().sum()
data.isnull().any(axis=1)
data[data.isnull().any(axis=1)]

data['BloodPressure'] = data['BloodPressure'].fillna(0)

data['BloodPressure'].fillna(0, inplace = True)

data['SkinThickness'].mean()
data['SkinThickness'] = data['SkinThickness'].fillna(20.564473684210526)

data['SkinThickness'] = data['SkinThickness'].fillna(data['SkinThickness'].mean())

data['SkinThickness'].fillna(data['SkinThickness'].mean(), inplace = True)


data['Insulin'] = data['Insulin'].fillna(data['Insulin'].median())

data = data.dropna()

class_counts = data.groupby('Class').size()
print(class_counts)

data.loc[data.Class == 'YES', 'Class'] = 1
data.loc[data.Class == 'NO', 'Class'] = 0

correlations = data.corr(method='pearson')
print(correlations)

import matplotlib.pyplot as plt
plt.hist(data.Glucose, bins = 15)
plt.show()

plt.plot(data.BloodPressure, data.Glucose)
plt.show()

plt.scatter(data.BloodPressure, data.Glucose )
plt.show()

plt.scatter(data.BloodPressure, data.Class )
plt.xlabel("BloodPressure")
plt.show()


plt.scatter(data.Glucose, data.Class )
plt.xlabel("Glucose")
plt.show()

plt.boxplot(data.BloodPressure)

plt.boxplot(data.Glucose)


# Short Form of Visualization
fig = plt.figure(figsize = (12.0,12.0))
ax = fig.gca()
data.boxplot(ax=ax)
plt.show()

fig = plt.figure(figsize = (12.0,12.0))
ax = fig.gca()
data.hist(ax=ax)
plt.show()

import seaborn as sns

sns.pairplot(data)

sns.pairplot(data,hue="Class")

cm = sns.heatmap(data.corr(), annot=True)

