import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import sklearn.tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
import sklearn.ensemble
from xgboost import XGBClassifier

#%%

df_label = pd.read_csv\
    ('/Users/katieforrest/Desktop/Python/Titanic_Spaceship/spaceship-titanic/train.csv')
df_test = pd.read_csv\
    ('/Users/katieforrest/Desktop/Python/Titanic_Spaceship/spaceship-titanic/test.csv')

df_train, df_val = \
    sk.model_selection.train_test_split(df_label, train_size =.8)

columns = ["Age", "RoomService", "VRDeck", "Spa", "FoodCourt", "ShoppingMall"]
X_train = df_train[columns]
y_train = df_train["Transported"]
X_val = df_val[columns]
y_val = df_val["Transported"]
X_test = df_test[columns]


impute_vals = {"HomePlanet": df_train["HomePlanet"].mode()}


X_train = X_train.fillna(impute_vals)
X_val = X_val.fillna(impute_vals)
X_test = X_test.fillna(impute_vals)

#%%
# Exploring the data
flags = (df_train['VIP'] == "True")
df_transported = df_train[flags]
df_not_transported = df_train[~flags]

plt.figure()
values,bins, temp = plt.hist(df_transported['Age'], density=True, 
                             alpha = .5, label = 'Transported', ec = 'black')
plt.hist(df_not_transported['Age'],bins=bins, density=True, 
                             alpha = .5, label = 'Not Transported', ec='black')
plt.legend()


# People that are more likely to survive:
    
    # Homeplanet
# younger people from earth >>
# Older people from Europa >>>
# 30-38 from mars >>>

    # Cabins
# A: older than 30
# B: Older than 20
# C: older than 20 younger than 62
# D: 29-42 and 49-55
# E: 21-35

# destinations
# PSO J318.5-22 age: 15-29


#%%
# Feature Engineering

# Implementaing valuable data that I discovered:
    
# VIP around age 30 are more likely to be transported
X_train["VIP_30"] = ((df_train["VIP"] == "True") & (df_train["Age"] > 27) \
                     & (df_train["Age"] < 35) ).astype('float')
X_val["VIP_30"] = ((df_val["VIP"] == "True") & (df_val["Age"] > 27) \
                     & (df_val["Age"] < 35) ).astype('float')
X_test["VIP_30"] = ((df_test["VIP"] == "True") & (df_test["Age"] > 27) \
                     & (df_test["Age"] < 35) ).astype('float')

# Younger people from earth are more likely to be transported
X_train["Earth_kids"] = ((df_train["HomePlanet"] == "Earth") \
                         & (df_train["Age"] < 22)).astype('float')
X_val["Earth_kids"] = ((df_val["HomePlanet"] == "Earth") \
                       & (df_val["Age"] < 22)).astype('float')
X_test["Earth_kids"] = ((df_test["HomePlanet"] == "Earth") \
                       & (df_test["Age"] < 22)).astype('float')
    

# Older people from europa are more likely to be transported
X_train["Europa_seniors"] = ((df_train["HomePlanet"] == "Europa") \
                             & (df_train["Age"] > 23)).astype('float')
X_val["Europa_seniors"] = ((df_val["HomePlanet"] == "Europa")\
                           & (df_val["Age"] > 23)).astype('float')
X_test["Europa_seniors"] = ((df_test["HomePlanet"] == "Europa")\
                           & (df_test["Age"] > 23)).astype('float')

# Cabins :)

# Cabin A
X_train['A'] = ((df_train['Cabin'].astype(str).str[0] == "A")\
                &(df_train['Age'] > 30)).astype('float')
X_val['A'] = ((df_val['Cabin'].astype(str).str[0] == "A")\
                &(df_val['Age'] > 30)).astype('float')
X_test['A'] = ((df_test['Cabin'].astype(str).str[0] == "A")\
                &(df_test['Age'] > 30)).astype('float')
    
# Cabin B
X_train['B'] = ((df_train['Cabin'].astype(str).str[0] == "B")\
                &(df_train['Age'] > 20)).astype('float')
X_val['B'] = ((df_val['Cabin'].astype(str).str[0] == "B")\
                &(df_val['Age'] > 20)).astype('float')
X_test['B'] = ((df_test['Cabin'].astype(str).str[0] == "B")\
                &(df_test['Age'] > 20)).astype('float')
    
# Cabin C
X_train['C'] = ((df_train['Cabin'].astype(str).str[0] == "C")\
                &(df_train['Age'] > 20) & (df_train['Age'] < 62)).astype('float')
X_val['C'] = ((df_val['Cabin'].astype(str).str[0] == "C")\
                &(df_val['Age'] > 20)& (df_val['Age'] < 62)).astype('float')
X_test['C'] = ((df_test['Cabin'].astype(str).str[0] == "C")\
                &(df_test['Age'] > 20)& (df_test['Age'] < 62)).astype('float')
    
# Cabin D
X_train['D'] = ((df_train['Cabin'].astype(str).str[0] == "D")\
                &(df_train['Age'] > 29) & (df_train['Age'] < 42)).astype('float')
X_val['D'] = ((df_val['Cabin'].astype(str).str[0] == "D")\
                &(df_val['Age'] > 29)& (df_val['Age'] < 42)).astype('float')
X_test['D'] = ((df_test['Cabin'].astype(str).str[0] == "D")\
                &(df_test['Age'] > 29)& (df_test['Age'] < 42)).astype('float')
    
# Cabin E
X_train['E'] = ((df_train['Cabin'].astype(str).str[0] == "E")\
                &(df_train['Age'] > 21) & (df_train['Age'] < 35)).astype('float')
X_val['E'] = ((df_val['Cabin'].astype(str).str[0] == "E")\
                &(df_val['Age'] > 21)& (df_val['Age'] < 35)).astype('float')
X_test['E'] = ((df_test['Cabin'].astype(str).str[0] == "E")\
                &(df_test['Age'] > 21)& (df_test['Age'] < 35)).astype('float')
    
# Destination
X_train["PSO"] = ((df_train['Destination'] == "PSO J318.5-22") \
                  & (df_train['Age'] > 15) & (df_train['Age'] < 29)).astype('float')
X_val["PSO"] = ((df_val['Destination'] == "PSO J318.5-22") \
                  & (df_val['Age'] > 15) & (df_val['Age'] < 29)).astype('float')
X_test["PSO"] = ((df_test['Destination'] == "PSO J318.5-22") \
                  & (df_test['Age'] > 15) & (df_test['Age'] < 29)).astype('float')
    
#%%

clf = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=.1,
    subsample=.8,
    colsample_bytree=.8)

#clf = sk.ensemble.RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
y_pred = y_pred == 1
num_correct = sum(y_pred == y_val)
acc_val = num_correct/len(X_val)
print(f'acc_val is: {acc_val}')

y_pred_test = clf.predict(X_test)
y_pred_test = y_pred_test == 1

dct = {"PassengerId": df_test['PassengerId'],
       "Transported": y_pred_test}

kaggle_sub = pd.DataFrame(dct)
kaggle_sub.to_csv('my_titanic_spaceship_submission_final!.csv', index=False)
