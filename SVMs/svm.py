import numpy as np
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

my_data = pd.read_csv('training.csv')
test = pd.read_csv('testing.csv')

my_data = my_data.drop(['Name'], axis=1).drop(['PassengerId'], axis=1).drop(['Cabin'], axis=1).drop(['Ticket'], axis=1)

my_data["Sex"] = np.where(my_data['Sex'] == 'male', 0, 1)

my_data.loc[(my_data['Embarked'] == 'S', 'Embarked')] = 1
my_data.loc[(my_data['Embarked'] == 'C', 'Embarked')] = 2
my_data.loc[(my_data['Embarked'] == 'Q', 'Embarked')] = 3

my_data = my_data.fillna(np.nan)
#sfdfsd
test = test.drop(['Name'], axis=1).drop(['PassengerId'], axis=1).drop(['Cabin'], axis=1).drop(['Ticket'], axis=1)

test["Sex"] = np.where(test['Sex'] == 'male', 0, 1)

test.loc[(test['Embarked'] == 'S', 'Embarked')] = 1
test.loc[(test['Embarked'] == 'C', 'Embarked')] = 2
test.loc[(test['Embarked'] == 'Q', 'Embarked')] = 3

test = test.fillna(np.nan)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

test = imp.fit_transform(test)
my_data = imp.fit_transform(my_data)

x = np.delete(my_data, 0, 1).astype(float)
y = [float(i[0]) for i in my_data]
a = test.astype(float)

# pd.option_context('display.max_rows', None, 'display.max_columns', None)
# np.set_printoptions(threshold=np.nan)# more options can be specified also
# f = open("stuff.txt", "w")
# print(my_data, file=f)
# Creating the hyperparameter grid 
  
# Instantiating logistic regression classifier 
param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],  
              'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001], 
              'kernel': ['rbf', 'sigmoid']}  
  
model = GridSearchCV(svm.SVC(), param_grid, cv=5, refit = True) 
  

#model = svm.SVC(gamma='scale')
model.fit(x, y)

b = model.predict(a)
b = np.array(b)

out_df = pd.DataFrame(columns=['solution'], data=b).astype(int)
out_df.index += 1
out_df.to_csv("new.csv")