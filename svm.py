from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('titanic.csv')
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

label_encoder = LabelEncoder()
df = df.apply(label_encoder.fit_transform)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df = imputer.fit_transform(df)


df = pd.DataFrame(df, columns=['PassengerId','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

y = df['Survived']
X = df.drop('Survived',axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

svc = SVC(kernel='linear', C=1)
svc.fit(X_train,y_train)

print("Accuracy on training set(without gridsearch): {:.3f}".format(svc.score(X_train,y_train)))
print("Accuracy on test set(without gridsearch): {:.3f}".format(svc.score(X_test,y_test)))
