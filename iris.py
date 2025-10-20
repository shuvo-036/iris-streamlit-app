#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt

import streamlit as st
# In[8]:


import seaborn as sns


# In[31]:


from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
#df['target'] = data.target
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)


# In[21]:


rows, cols = df.shape
print("Rows:" , rows)
print("colums:" , cols)


# In[37]:


y=df["species"]
y


# In[40]:


x=df.drop("species" , axis=1)
x


# In[41]:


y


# In[43]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)


# In[45]:


x_train


# In[50]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)   # create the model
knn.fit(x_train, y_train)                   # train the model


# In[51]:


knn5 = KNeighborsClassifier(n_neighbors=3)
knn5.fit(x_train, y_train)
accuracy = knn5.score(x_test, y_test)
print("Accuracy with 3 neighbors:", accuracy)


# In[54]:


y_pred = knn.predict(x_test)
print(y_pred)


# In[55]:


x_test.head()


# In[70]:


newdata = pd.DataFrame({
    "sepal length (cm)": [1.2],
    "sepal width (cm)": [1.5],
    "petal length (cm)": [2.2],
    "petal width (cm)": [0.6]
})
prediction= knn.predict(newdata)
prediction[0]


# In[ ]:




