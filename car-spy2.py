# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:01:17 2024

@author: AAC
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree



#%%
df = pd.read_csv('car_evaluation.csv', header=None)

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

#%%
for col in col_names:
    print(df[col].value_counts())

#%%
x = df.drop(['class'], axis= 1)
y = df['class']

#%%
df.head()
df.shape

#data is preprocessed. There are no missing values or anomalies
df.info()
#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state= 42)
#%% Mapping the columns

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
                            mapping=[{'col': 'buying', 'mapping': {None: 0, 'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}},
                                      {'col': 'maint', 'mapping': {None: 0, 'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}},
                                      {'col': 'doors', 'mapping': {None: 0, '2': 2, '3': 3, '4': 4, '5more': 5}},
                                      {'col': 'persons', 'mapping': {None: 0, '2': 2, '4': 4, 'more': 5}},
                                      {'col': 'lug_boot', 'mapping': {None: 0, 'small': 1, 'med': 2, 'big': 3}},
                                      {'col': 'safety', 'mapping': {None: 0, 'low': 1, 'med': 2, 'high': 3}}]
                            )

#%%

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)

#%% Assigning the values for rfc

rfc = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=7)

# When creating an object the random forests randomly creates forests and do bootstrapping 
# Thats why you have to give Random State or else the bagging will be done differently in all of the files who run the code
# Another parameter how many trees you want to make that is what nestimators
# Basically you have to try and set the random_state but why stop at 100 and not more. The

rfc.fit(x_train, y_train)
#training the data
y_pred = rfc.predict(x_test)
#we need the labels on the test data
#%%
print(f'Model Accuracy : {accuracy_score(y_test, y_pred)}')

#%%

cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix Shape: {cm.shape}')
print(f'Confusion Matrix: \n{cm}')

#%%
#some datasets will show best results in entropy and some in gini index

plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=rfc.classes_, yticklabels=rfc.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#%%

plt.figure(figsize=(160,80))
plot_tree(rfc.estimators_[5], feature_names = list(x_test.columns),class_names=['unacc','acc','good','vgood'],filled=True, max_depth=2);
plt.show()

#%%

dfo = pd.DataFrame(y_pred, columns=['predictions'])
dfo.to_csv('prediction.csv', index=False)
