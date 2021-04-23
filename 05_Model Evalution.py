from pandas import read_csv
import pandas as pd

data = read_csv('C:/Users/NIRANJAN/Downloads/Intern Jun 2019/Datasets/14_Diabetes/Diabetes.csv')

data.columns = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Class')

X = data.iloc[:, [0,1,2,3,4,5,6,7]]
y = data.iloc[:, 8]

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())
print(results.std())

from sklearn.tree import DecisionTreeClassifier
kfold = KFold(n_splits=10, random_state=7)
model = DecisionTreeClassifier()
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())

from sklearn.neighbors import KNeighborsClassifier
kfold = KFold(n_splits=10, random_state=7)
model = KNeighborsClassifier()
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())

from sklearn.naive_bayes import GaussianNB
kfold = KFold(n_splits=10, random_state=7)
model = GaussianNB()
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())

from sklearn.svm import SVC
kfold = KFold(n_splits=10, random_state=7)
model = SVC(gamma = 'auto')
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())

from sklearn.ensemble import RandomForestClassifier
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=500, max_features=4)
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())

from sklearn.ensemble import AdaBoostClassifier
kfold = KFold(n_splits=10, random_state=7)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
print(results)
print(results.mean())

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF', RandomForestClassifier(n_estimators=500, max_features=4)))
models.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)))


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring= 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
    
import matplotlib.pyplot as pyplot    
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

