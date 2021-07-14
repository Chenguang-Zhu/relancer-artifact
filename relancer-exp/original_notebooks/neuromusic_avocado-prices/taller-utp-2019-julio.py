#!/usr/bin/env python
# coding: utf-8

# # Taller de introduccion a inteligencia artificial UTP 2019 - Julio
# 
# En este taller aprenderemos a utilizar las librerias mas utilizadas en los modelos de prediccion, ademas de conocer un poco  sobre los modelo de clasificacion 
# 
# Autor: Miguel Angel Cotrina Espinoza

# ## Cargamos nuestras librerias
# -------------------------------------------
# - numpy 
# <a href="https://www.numpy.org"><img src="https://www.numpy.org/_static/numpy_logo.png" width="250"></a>
# -----------------------------------------
# 
# - pandas 
# <a href="https://pandas.pydata.org"><img src="https://pandas.pydata.org/_static/pandas_logo.png" width="250"></a>
# -----------------------------------------
# 
# - matplotlib 
# <a href="https://matplotlib.org "><img src="https://matplotlib.org/_static/logo2.png" width="250"></a>
# --------------------------------------------
# 
# - sklearn (Science Kit Learn) 
# <a href="https://scikit-learn.org/stable"><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width="250"></a>
# ----------------------------------------

# In[ ]:


#Principales librerias para la carga y preproceso de datos

import numpy as np # Libreria de algebra lineal
import pandas as pd # Libreria para manipulacion de datos
import matplotlib.pyplot as plt # matplotlib libreria para generar graficos y pyplot premite una interfaz como matlab


# In[ ]:


#Libreria de split

from sklearn.model_selection import train_test_split # X_train,X_test,y_train,y_test 


# In[ ]:


#Principales librerias para realizar modelos de prediccion

#Regresion
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#Principales librerias de metricas

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


# ## Para poder realizar nuestro primer modelo, vamos a trabajar los siguientes paso
# 
# - cargar los datos ok
# - Ingenieria de caracteristicas
# - Seleccion de caracteristicas 
# - Dividir los datos Train / Test ok
# - selecciones del modelo ok
# - Entrenar el modelo ok
# - Generar nuestros valores de prediccion ok
# - Calcular las metricas ok

# In[ ]:


datos = pd.read_csv("../../../input/neuromusic_avocado-prices/avocado.csv")
publicidad = pd.read_csv("../../../input/neuromusic_avocado-prices/avocado.csv",sep=";")


# In[ ]:


publicidad.head(5)


# In[ ]:


import seaborn as sns
sns.set(font_scale = 1.5)
corr = publicidad.corr('spearman') 
plt.figure(figsize = ( 5 , 5 )) 
print()


# In[ ]:


publicidad.shape


# In[ ]:


y = publicidad.iloc[:,0]
x =publicidad.iloc[:,1:4]


# In[ ]:





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


lr = LinearRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred = lr.predict(X_test)


# In[ ]:


print('Variance score: %.2f' % mean_squared_error(y_test, y_pred))


# In[ ]:


print()
sns.set(font_scale = 1)
sns.distplot(y);


# In[ ]:




