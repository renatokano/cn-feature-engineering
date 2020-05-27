#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv", decimal=',')


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.dtypes


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[5]:


def q1():
    region_names = list(set(countries['Region'].str.strip()))
    region_names.sort()
    return region_names


# In[6]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[7]:


discretizer = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
pop_density_bins = discretizer.fit_transform(countries[['Pop_density']])


# In[8]:


def q2():
    return int((pop_density_bins.flatten().astype(int) == 9).sum())


# In[9]:


q2()


# ## Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[10]:


n_regions = len(set(countries['Region']))
# climate has missing values
n_climates = len(set(countries['Climate'].fillna(0)))


# In[11]:


def q3():
    return (n_regions + n_climates)


# In[12]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[13]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[14]:


countries.head()


# In[15]:


# select o/ int64/float64 columns (features)
numeric_countries_df = countries.select_dtypes(include=np.number)
# get all missing values from these columns
numeric_countries_df.isna().sum()


# In[16]:


# create a new pipeline: 
# 1st: imputation transformer for completing missing values using the 'median' 
# 2nd: standardize features
numeric_pipeline = Pipeline(steps=[
    ('median_imputer', SimpleImputer(strategy='median')),
    ('standardize_features', StandardScaler())
])


# In[17]:


# create a new dataframe w/ the same columns of the 'countries' dataframe
df_test_country = pd.DataFrame([test_country], columns=countries.columns)
df_test_country


# In[18]:


numeric_countries = list(numeric_countries_df.columns)


# In[19]:


def q4():
    # fit the data (j/ numeric features)
    pipeline_fit = numeric_pipeline.fit(countries[numeric_countries])
    # transform train data
    pipeline_train_transform = pipeline_fit.transform(countries[numeric_countries])
    # get a column index
    arable = numeric_countries.index('Arable')
    # transform test data and return the 'Arable' value
    return float(pipeline_fit.transform(df_test_country[numeric_countries])[0][arable].round(3))


# In[20]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[21]:


# get 'Net_migration' feature
net_migration = countries['Net_migration']
# calculate q1, q3, iqr
quartile1 = net_migration.quantile(0.25)
quartile3 = net_migration.quantile(0.75)
iqr = quartile3 - quartile1
# calculate the outlier intervale
non_outlier_interval_iqr = [quartile1 - 1.5 * iqr, quartile3 + 1.5 * iqr]
# outliers
inf_outliers = net_migration[(net_migration < non_outlier_interval_iqr[0])].count()
sup_outliers = net_migration[(net_migration > non_outlier_interval_iqr[1])].count()


# In[22]:


sns.boxplot(net_migration, orient='vertical');


# In[31]:


def q5():
    return (int(inf_outliers), int(sup_outliers), False)


# In[39]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[25]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[40]:


def q6():
    # convert a collection of text documents to a matrix of token counts
    count_vectorizer = CountVectorizer()
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
    # get the phone index
    phone_idx = count_vectorizer.vocabulary_.get('phone')
    return int(newsgroups_counts[:,phone_idx].sum())


# In[27]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[51]:


def q7():
    # convert a collection of raw documents to a matrix of TF-IDF features
    tfidf_vectorizer = TfidfVectorizer()
    newsgroups_tfidf_vectorized = tfidf_vectorizer.fit_transform(newsgroups.data)
    # get the phone index
    phone_idx = tfidf_vectorizer.vocabulary_.get('phone')
    return float(newsgroups_tfidf_vectorized[:,phone_idx].sum().round(3))


# In[52]:


q7()


# In[ ]:




