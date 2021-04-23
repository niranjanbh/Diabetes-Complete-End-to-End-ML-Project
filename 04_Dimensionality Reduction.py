from pandas import read_csv
import pandas as pd

data = read_csv('C:/Users/NIRANJAN/Downloads/Intern Jun 2019/Datasets/14_Diabetes/Diabetes.csv')

data.columns = ('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 
                'DiabetesPedigreeFunction', 'Age', 'Class')

data.loc[data.Class == 'YES', 'Class'] = 1
data.loc[data.Class == 'NO', 'Class'] = 0

X = data.iloc[:, [0,1,2,3,4,5,6,7]]
y = data.iloc[:, 8]


#Chi square test
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select_feature = SelectKBest(chi2, k=4).fit(X, y)
selected_features_df = pd.DataFrame({'Feature':list(X.columns),
                                    'Scores':select_feature.scores_})
selected_features_df.sort_values(by='Scores', ascending=False)
X_chi = select_feature.transform(X)



#Recursive feature elimination with cross validation
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
rfecv = RFECV(estimator=LogisticRegression(solver='liblinear'), step=1, scoring='accuracy')
rfecv = rfecv.fit(X, y)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', X.columns[rfecv.support_])
rfecv.grid_scores_

import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




#PCA
from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=8)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)




#Feature Importence
from sklearn.ensemble import RandomForestClassifier
# feature extraction
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
print(model.feature_importances_)
