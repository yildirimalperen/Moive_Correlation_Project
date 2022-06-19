#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # adjusting the configuration of the plots we will create


# In[2]:


# read  in the data

df = pd.read_csv(r'C:\Users\aalpe\OneDrive\Masaüstü\Work\Projects\Python\Portfolio_Project\movies_dataset.csv')


# In[3]:


#lets look at the data
df


# In[4]:


# lets see if there is any missing data

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,pct_missing))


# In[5]:


# ddata types for our colums
df.dtypes


# In[6]:


df.budget = df.budget.fillna(0)
df.budget.isnull().sum()
df.budget = df.budget.astype('int64')


# In[7]:


df.gross = df.gross.fillna(0)
df.gross.isnull().sum()
df.gross = df.gross.astype('int64')


# In[108]:


def myfunction():
    for i in range(7668):
        print(str(df.released[i]))
 


# In[26]:


df.sort_values(by=['gross'],inplace = False, ascending = False)


# In[25]:


pd.set_option('display.max_rows',None)


# In[8]:


#Drop any duplicates
df['company'].drop_duplicates().sort_values(ascending=False)


# In[34]:


pd.reset_option('all')


# In[9]:


# Budget high correlation
# Company high correlation
# Scatter plot with budget vs gross revenue

plt.scatter(x=df['budget'],y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Budget for Film')
plt.ylabel('Gross Earnings')
plt.show()



# In[10]:


#Plot budget vs gross using seaborn

sns.regplot(x='budget',y='gross',data=df, scatter_kws={'color':'purple'},line_kws={'color':'brown'})


# In[11]:


#lets start looking at correlation
df.corr(method='pearson') #pearson, kendall, spearman methods are available to determain correlation


# In[15]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[16]:


#Looks at comppanies
df.head()


# In[20]:


df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes

df_numerized
        


# In[21]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix for Numeric Features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[22]:


df_numerized.corr()


# In[26]:


correlation_mat = df_numerized.corr()
corr_pairs = correlation_mat.unstack()
corr_pairs.sort_values()

sorted_pairs = corr_pairs.sort_values()


# In[30]:


high_corr = sorted_pairs[(sorted_pairs) > 0.4]
high_corr
#score, votes and budget have the highest correlation to gross earnings 

