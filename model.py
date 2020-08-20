#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# import bins data

bins = pd.read_excel("UpdatedBinData.xlsx")
bins.head()


# In[3]:


bins.shape


# In[4]:


# drop columns not being used

bins.drop(['Name', 'Date', 'Prev Bin', 'Name'], axis = 1, inplace = True)
bins.rename({'Retailer':'Retailer #'}, axis=1, inplace=True)
bins.sample(5)


# In[5]:


bins.sample(5)


# In[6]:


# import retailers, sales, and scratchers data

dff = pd.read_csv('Retailers_Sales_Scratchers_Joined.csv', index_col = 0)
dff.sample(5)


# In[7]:


# create GGR column using conditions of sales > 0 or sales <= 0

dff.loc[dff['Scratcher Sales'] > 0, 'GGR'] = dff['Scratcher Sales'] * (1 - 0.065) * (1 - dff['Payout %'])

dff.loc[dff['Scratcher Sales'] <= 0, 'GGR'] = dff['Scratcher Sales']


# In[8]:


# drop columns not being used

dff.drop(['Week End Date','Retailer Name', 'Game Name', 'Street1', 'Street2', 'City', 
          'Organization code', 'Business Code', 'Zip Mean Income', 'Pk Val', 'Game Odds', 
          'Payout %', 'Max LT Winners', 'Top Prize', 'Top Prize Ct', 'Top Prizes Cashed', 
          'Validation Status', 'Launch Dt', 'End Dist Dt', 'End Valid Dt'], 
         axis = 1, inplace = True)

dff.sample(5)


# In[9]:


# merge bins and retailer/sales data

dff = pd.merge(dff, bins, how = 'left', left_on = 'Retailer', right_on = 'Retailer #')
dff.drop(['Retailer #'], axis = 1, inplace = True)
dff.sample(5)


# In[10]:


dff.isnull().sum()


# In[11]:


# drop missing values (inactive retailers and retailers without bin data recorded)

dff.dropna(subset=['Zipcode'], inplace = True)
dff.dropna(subset=['Business Type'], inplace = True)
dff.dropna(subset=['Bin Size'], inplace = True)


# In[12]:


dff.isnull().sum()


# In[13]:


# fill missing demographic data

from sklearn.impute import SimpleImputer

impInc = SimpleImputer(missing_values = np.nan, strategy = 'mean')

dfftemp = pd.DataFrame(impInc.fit_transform(dff[['Zip Median Income']]), index=dff.index, columns = ['Zip Median Inc'])
dff = pd.concat([dff, dfftemp], axis = 1)
dff.drop('Zip Median Income', axis = 1, inplace = True)

dff.sample(5)


# In[14]:


dff.isnull().sum()


# In[15]:


# fill missing demographic data

from sklearn.impute import SimpleImputer

impPop = SimpleImputer(missing_values = np.nan, strategy = 'mean')

dfftemp2 = pd.DataFrame(impPop.fit_transform(dff[['Zip Population']]), index=dff.index, columns = ['Zip Pop'])
dff = pd.concat([dff, dfftemp2], axis = 1)
dff.drop('Zip Population', axis = 1, inplace = True)

dff.sample(5)


# In[16]:


dff.isnull().sum()


# In[17]:


# get license year from license date column

dff['License Year'] = pd.DatetimeIndex(dff['License Date']).year
dff.sample(5)


# In[18]:


# calculate license length by finding difference between this year and license year

dff['License Length'] = 2020 - dff['License Year']
dff.sample(5)


# In[19]:


# drop columns not needed

dff.drop(['License Date'], axis = 1, inplace = True)
dff.sample(5)


# In[20]:


# creating calculated field for number of packs ordered

dff['Packs Ordered'] = dff['Scratcher Sales'] / (dff['Pk Size'] * dff['Price'])
dff['Packs Ordered'].sample(5)


# In[21]:


# examine play style attributes

dff['Play Style'].value_counts()


# In[22]:


pivot_style = pd.pivot_table(dff, values = 'Packs Ordered', index = 'Retailer', 
                             columns = 'Play Style', aggfunc = 'mean', fill_value = 0)
pivot_style.head()


# In[23]:


# examine theme attributes

dff['Theme'].value_counts()


# In[24]:


pivot_theme = pd.pivot_table(dff, values = 'Packs Ordered', index = 'Retailer', 
                             columns = 'Theme', aggfunc = 'mean', fill_value = 0)
pivot_theme.head()


# In[25]:


# pivot table to create average packs ordered per week

pivot_avg_packs = pd.pivot_table(dff, values = 'Packs Ordered', index = 'Retailer', 
                                 columns = 'Game Number', aggfunc = 'mean', fill_value = 0)
pivot_avg_packs.head()


# In[26]:


dff.sample(5)


# In[27]:


# build pivot table to aggregate total sales, average ticket price, count of theme and play style, count of games

pivot_sales = pd.pivot_table(dff, 
                             index = 'Retailer', 
                             values = ['Scratcher Sales', 'Price', 'Play Style', 'Theme', 
                                       'Game Number', 'GGR'], 
                             aggfunc = {'Scratcher Sales':np.sum, 
                                        'Price':np.mean, 
                                        'Play Style':pd.Series.nunique, # 16 unique styles
                                        'Theme':pd.Series.nunique,      # 15 unique themes
                                        'Game Number':pd.Series.nunique, 
                                        'GGR':np.sum})  

pivot_sales.sample(5)


# In[28]:


pivot_sales.shape


# In[29]:


# build dataframe with other information about retailers

other_data = dff[['Retailer', 'Zipcode', 'Bin Size', 'License Length', 
                  'Zip Median Inc', 'Zip Pop', 'Business Type', 
                  'Organization Type']].sort_values('Retailer')
other_data.shape


# In[30]:


other_data.drop_duplicates('Retailer', inplace = True)
other_data.shape


# In[31]:


other_data.sample(5)


# In[32]:


# one-hot encode business type and organization type categorical variables

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categories = 'auto', sparse = False, dtype = int)

catvars = ['Business Type']

dfcat = pd.DataFrame(ohe.fit_transform(other_data[catvars]), 
                     columns = ohe.get_feature_names(), 
                     index = other_data.index)

other_data = pd.concat([other_data, dfcat], axis = 1)
other_data.drop(catvars, axis = 1, inplace = True)


# In[33]:


other_data.sample(5)


# In[34]:


# merge retailer attribute data with aggregate data

other_data = pd.merge(other_data, pivot_sales, left_on = 'Retailer', right_on = 'Retailer')
finalDF = other_data.copy()
finalDF.sample(5)


# In[35]:


finalDF['Active length'] = finalDF['License Length'].apply(lambda x:2 if x >= 2 else x)


# In[36]:


finalDF['Sales per Year'] = finalDF['Scratcher Sales'] / finalDF['Active length']
finalDF.sample(5)


# In[37]:


finalDF['GGR per Year'] = finalDF['GGR'] / finalDF['Active length']
finalDF.sample(5)


# In[38]:


finalDF['GGR per Year per Bin'] = (finalDF['GGR'] / finalDF['Active length'])/finalDF['Bin Size']
finalDF.sample(5)


# In[39]:


finalDF = finalDF.set_index('Retailer')
finalDF.sample(5)


# In[40]:


finalDF['Sales per Person'] = finalDF['Scratcher Sales'] / finalDF['Zip Pop']
finalDF.sample(5)


# In[41]:


finalDF['Play Style Variety %'] = finalDF['Play Style'].apply(lambda x:x/16)
finalDF['Theme Variety %'] = finalDF['Theme'].apply(lambda x:x/15)
finalDF.sample(5)


# In[42]:


finalDF.drop(['Play Style', 'Theme', 'Scratcher Sales', 'GGR'], axis = 1, inplace = True)


# In[43]:


finalDF.rename(columns = {'Price':'Average Ticket Price', 'Game Number':'Count Games Offered'}, inplace = True)
finalDF.sample(5)


# In[44]:


finalDF['Sales per Year per Person'] = finalDF['Sales per Year'] / finalDF['Zip Pop']
finalDF.sample(5)


# In[45]:


income_mean = np.mean(finalDF['Zip Median Inc'])
finalDF['Scaled Median Income'] = finalDF['Zip Median Inc'].apply(lambda x:(1 + ((x - income_mean)/income_mean)))
finalDF.sample(5)


# In[46]:


pop_mean = np.mean(finalDF['Zip Pop'])
finalDF['Scaled Population'] = finalDF['Zip Pop'].apply(lambda x:(1 + ((x - pop_mean)/pop_mean)))
finalDF.sample(5)


# In[47]:


from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
sll = pd.DataFrame(mms.fit_transform(finalDF[['License Length']]), 
                   columns = ['Scaled License Length'], index = finalDF.index)
finalDF = pd.concat([finalDF, sll], axis = 1)

finalDF.sample(5)


# In[48]:


clusterDf = finalDF.drop(['Zip Median Inc', 'Zip Pop', 'Count Games Offered', 
                          'Sales per Year', 'Sales per Person', 'Play Style Variety %', 
                          'Theme Variety %', 'Sales per Year per Person', 'GGR per Year per Bin','GGR per Year',
                          'Organization Type', 'Zipcode', 'License Length', 
                          'Active length','Average Ticket Price'], 
                       axis = 1)

sales_info = finalDF[['Zipcode', 'Zip Median Inc', 'Zip Pop', 
                      'Count Games Offered', 'Sales per Year', 'Sales per Person', 
                      'Play Style Variety %', 'Theme Variety %', 'GGR per Year per Bin','GGR per Year',
                      'Sales per Year per Person', 'License Length', 
                      'Active length','Average Ticket Price']]


# In[49]:


clusterDf.sample(10)


# ### KMeans Clustering

# In[50]:


from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

errorlst = pd.DataFrame(data = None, columns = ['k', 'error'])

for k in range(2, 10):
    km = KMeans(n_clusters = k)
    
    cluster_labels = km.fit_predict(clusterDf)
    silhouette_avg = silhouette_score(clusterDf, cluster_labels)

    error = km.inertia_  #Sum of squared distances of samples to their closest cluster center
    errorlst = errorlst.append({'k':k, 'error':error}, ignore_index = True)
    
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)    

plt.plot(errorlst['k'], errorlst['error'], 'o-')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters k')
plt.show()


# In[51]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters = 4, random_state = 0)

clusters = km.fit_predict(clusterDf)

clust = pd.DataFrame(clusters, columns = ['KMeans L1'], index = clusterDf.index)
clust.sample(5)


# In[52]:


clust['KMeans L1'].value_counts()


# In[53]:


print("Silhouette: ", metrics.silhouette_score(clusterDf, clusters))


# In[54]:


clusterDf = pd.concat([clusterDf, clust], axis = 1)
clusterDf.head(10)


# ### Join clustering results with sales information for second-level analysis in Tableau

# In[55]:


data = pd.merge(clusterDf, sales_info, how = 'right', left_index = True, right_index = True)
data.head()


# In[56]:


# build dataframe with categorical information about retailers

categories = dff[['Retailer', 'Business Type', 'Organization Type']].sort_values('Retailer')
categories.shape


# In[57]:


categories.drop_duplicates('Retailer', inplace = True)
categories.shape


# In[58]:


categories = categories.set_index('Retailer')
categories.sample(5)


# In[59]:


data = pd.merge(data, categories, how = 'left', left_index = True, right_index = True)
data.head()


# In[60]:


data.drop(['x0_Auto Service / Gas Stations', 'x0_Bars and Restaurants',
       'x0_Bowling Centers',
       'x0_Chain Convenience Stores (Circle K / 7-Eleven)',
       'x0_Chain Supermarkets', 'x0_Drug Stores / Pharmacies',
       'x0_Independent Convenience Stores', 'x0_Independent Supermarkets',
       'x0_Liquor Stores', 'x0_Shopping Mall',
       'x0_Smoke/Gift Shops/News Stands', 'x0_Specialty Non-Grocery - Misc',
       'x0_Truck Service Centers'], axis = 1, inplace = True)


# In[61]:


data.head()


# # 2nd cluster 

# In[62]:


pivot_style.head()


# In[63]:


pivot_style.columns = pivot_style.columns.str.strip()

pivot_style.rename(columns = {'ADD UP':'Style ADD UP', 
                        'BINGO':'Style BINGO', 
                        'COORDINATE':'Style COORDINATE', 
                        'CROSSWORD':'Style CROSSWORD', 
                        'EXTENDED PLAY':'Style EXTENDED PLAY', 
                        'FIND NUMBER':'Style FIND NUMBER', 
                        'FIND SYMBOL':'Style FIND SYMBOL', 
                        'KEY NUMBER MATCH':'Style KEY NUMBER MATCH', 
                        'KEY SYMBOL MATCH':'Style KEY SYMBOL MATCH', 
                        'MATCH 2':'Style MATCH 2', 
                        'MATCH 3':'Style MATCH 3', 
                        'MAZE':'Style MAZE', 
                        'MULTIPLE':'Style MULTIPLE', 
                        'OTHER':'Style OTHER', 
                        'SLINGO':'Style SLINGO', 
                        'TIC-TAC-TOE':'Style TIC-TAC-TOE'}, 
            inplace = True)


# In[64]:


pivot_theme.head()


# In[65]:


pivot_theme.columns = pivot_theme.columns.str.strip()

pivot_theme.rename(columns = {'ARIZONA':'Theme ARIZONA', 
                        'BINGO':'Theme BINGO', 
                        'CARDS':'Theme CARDS', 
                        'CROSSWORD':'Theme CROSSWORD', 
                        'GAMING':'Theme GAMING', 
                        'LICENSED PROPERTY':'Theme LICENSED PROPERTY',
                        'LUCK':'Theme LUCK', 
                        'LUCKY NUMBERS':'Theme LUCKY NUMBERS', 
                        'MONEY':'Theme MONEY', 
                        'OTHER':'Theme OTHER', 
                        'SEASONAL':'Theme SEASONAL', 
                        'SLINGO':'Theme SLINGO',
                        'SPECIALTY':'Theme SPECIALTY', 
                        'SPORTS':'Theme SPORTS', 
                        'WHIMSICAL':'Theme WHIMSICAL'}, 
            inplace = True)


# In[66]:


ClusterDf_purchasing = pd.merge(data[['KMeans L1', 'GGR per Year']], 
                                pivot_style, 
                                how = 'left', 
                                left_on = 'Retailer', 
                                right_on = 'Retailer')

ClusterDf_purchasing.head()


# In[67]:


ClusterDf_purchasing = pd.merge(ClusterDf_purchasing, 
                                pivot_theme, 
                                how = 'left', 
                                left_on = 'Retailer', 
                                right_on = 'Retailer')

ClusterDf_purchasing.head()


# In[68]:


# Segment for first level

df_segment0 = ClusterDf_purchasing[ClusterDf_purchasing['KMeans L1'] == 0]
df_segment1 = ClusterDf_purchasing[ClusterDf_purchasing['KMeans L1'] == 1]
df_segment2 = ClusterDf_purchasing[ClusterDf_purchasing['KMeans L1'] == 2]
df_segment3 = ClusterDf_purchasing[ClusterDf_purchasing['KMeans L1'] == 3]


# In[69]:


df_segment0.head()


# In[70]:


# drop cluster for first level

df_segment0 = df_segment0.drop(['KMeans L1'], axis = 1)
df_segment1 = df_segment1.drop(['KMeans L1'], axis = 1)
df_segment2 = df_segment2.drop(['KMeans L1'], axis = 1)
df_segment3 = df_segment3.drop(['KMeans L1'], axis = 1)


# In[71]:


print(df_segment0.shape)
print(df_segment1.shape)
print(df_segment2.shape)
print(df_segment3.shape)


# In[72]:


### 2nd level cluster ###
#########################

# cluster 0 for first level cluster

errorlst = pd.DataFrame(data = None, columns = ['k', 'error'])

for k in range(2, 10):
    km = KMeans(n_clusters = k)
    
    cluster_labels = km.fit_predict(df_segment0)
    silhouette_avg = silhouette_score(df_segment0, cluster_labels)

    error = km.inertia_  #Sum of squared distances of samples to their closest cluster center
    errorlst = errorlst.append({'k':k, 'error':error}, ignore_index = True)
    
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)    

plt.plot(errorlst['k'], errorlst['error'], 'o-')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters k')
plt.show()


# In[73]:


### 2nd level cluster ###
#########################

# cluster 1 for first level cluster

errorlst = pd.DataFrame(data = None, columns = ['k', 'error'])

for k in range(2, 10):
    km = KMeans(n_clusters = k)
    
    cluster_labels = km.fit_predict(df_segment1)
    silhouette_avg = silhouette_score(df_segment1, cluster_labels)

    error = km.inertia_  #Sum of squared distances of samples to their closest cluster center
    errorlst = errorlst.append({'k':k, 'error':error}, ignore_index = True)
    
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)    

plt.plot(errorlst['k'], errorlst['error'], 'o-')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters k')
plt.show()


# In[74]:


### 2nd level cluster ###
#########################

# cluster 2 for first level cluster

errorlst = pd.DataFrame(data = None, columns = ['k', 'error'])

for k in range(2, 10):
    km = KMeans(n_clusters = k)
    
    cluster_labels = km.fit_predict(df_segment2)
    silhouette_avg = silhouette_score(df_segment2, cluster_labels)

    error = km.inertia_  #Sum of squared distances of samples to their closest cluster center
    errorlst = errorlst.append({'k':k, 'error':error}, ignore_index = True)
    
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)    

plt.plot(errorlst['k'], errorlst['error'], 'o-')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters k')
plt.show()


# In[75]:


### 2nd level cluster ###
#########################

# cluster 3 for first level cluster

errorlst = pd.DataFrame(data = None, columns = ['k', 'error'])

for k in range(2, 10):
    km = KMeans(n_clusters = k)
    
    cluster_labels = km.fit_predict(df_segment3)
    silhouette_avg = silhouette_score(df_segment3, cluster_labels)

    error = km.inertia_  #Sum of squared distances of samples to their closest cluster center
    errorlst = errorlst.append({'k':k, 'error':error}, ignore_index = True)
    
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)    

plt.plot(errorlst['k'], errorlst['error'], 'o-')
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters k')
plt.show()


# In[76]:


# KMEANS FOR SECOND LEVEL

km = KMeans(n_clusters = 3, random_state = 0)


# In[77]:


# Segment 0

kmeans20 = km.fit_predict(df_segment0)
df_seg0_2 = pd.DataFrame(kmeans20, columns = ['KMeans L2'], index = df_segment0.index)

df_seg0_2.head()


# In[78]:


df_seg0_2['KMeans L2'].value_counts()


# In[79]:


# Segment 1

kmeans21 = km.fit_predict(df_segment1)
df_seg1_2 = pd.DataFrame(kmeans21, columns = ['KMeans L2'], index = df_segment1.index)

df_seg1_2.head()


# In[80]:


df_seg1_2['KMeans L2'].value_counts()


# In[81]:


# Segment 2

kmeans22 = km.fit_predict(df_segment2)
df_seg2_2 = pd.DataFrame(kmeans22, columns = ['KMeans L2'], index = df_segment2.index)

df_seg2_2.head()


# In[82]:


df_seg2_2['KMeans L2'].value_counts()


# In[83]:


# Segment 3

kmeans23 = km.fit_predict(df_segment3)
df_seg3_2 = pd.DataFrame(kmeans23, columns = ['KMeans L2'], index = df_segment3.index)

df_seg3_2.head()


# In[84]:


df_seg3_2['KMeans L2'].value_counts()


# In[85]:


clustl2 = pd.concat([df_seg0_2, df_seg1_2, df_seg2_2, df_seg3_2]).sort_values('Retailer')
clustl2.head()


# In[86]:


data.head()


# In[87]:


data = pd.merge(data, clustl2, how = 'left', left_index = True, right_index = True)
data.head()


# In[88]:


dff.sample(10)


# In[89]:


dff = pd.merge(dff, clustl2, how = 'left', left_on = 'Retailer', right_on = 'Retailer')


# In[90]:


dff = pd.merge(dff, clust, how = 'left', left_on = 'Retailer', right_on = 'Retailer')
dff.head()


# In[91]:


dff.to_csv("FinalResultsUnagg.csv")


# In[92]:


style = pd.pivot_table(data = dff, index = 'Retailer', columns = 'Play Style', 
                       values = 'GGR', aggfunc = np.sum)

style.head()


# In[93]:


style.columns = style.columns.str.strip()
style.columns


# In[94]:


style.rename(columns = {'ADD UP':'Style ADD UP', 
                        'BINGO':'Style BINGO', 
                        'COORDINATE':'Style COORDINATE', 
                        'CROSSWORD':'Style CROSSWORD', 
                        'EXTENDED PLAY':'Style EXTENDED PLAY', 
                        'FIND NUMBER':'Style FIND NUMBER', 
                        'FIND SYMBOL':'Style FIND SYMBOL', 
                        'KEY NUMBER MATCH':'Style KEY NUMBER MATCH', 
                        'KEY SYMBOL MATCH':'Style KEY SYMBOL MATCH', 
                        'MATCH 2':'Style MATCH 2', 
                        'MATCH 3':'Style MATCH 3', 
                        'MAZE':'Style MAZE', 
                        'MULTIPLE':'Style MULTIPLE', 
                        'OTHER':'Style OTHER', 
                        'SLINGO':'Style SLINGO', 
                        'TIC-TAC-TOE':'Style TIC-TAC-TOE'}, 
            inplace = True)


# In[95]:


theme = pd.pivot_table(data = dff, index = 'Retailer', columns = 'Theme', 
                       values = 'GGR', aggfunc = np.sum)

theme.head()


# In[96]:


theme.columns = theme.columns.str.strip()
theme.columns


# In[97]:


theme.rename(columns = {'ARIZONA':'Theme ARIZONA', 
                        'BINGO':'Theme BINGO', 
                        'CARDS':'Theme CARDS', 
                        'CROSSWORD':'Theme CROSSWORD', 
                        'GAMING':'Theme GAMING', 
                        'LICENSED PROPERTY':'Theme LICENSED PROPERTY',
                        'LUCK':'Theme LUCK', 
                        'LUCKY NUMBERS':'Theme LUCKY NUMBERS', 
                        'MONEY':'Theme MONEY', 
                        'OTHER':'Theme OTHER', 
                        'SEASONAL':'Theme SEASONAL', 
                        'SLINGO':'Theme SLINGO',
                        'SPECIALTY':'Theme SPECIALTY', 
                        'SPORTS':'Theme SPORTS', 
                        'WHIMSICAL':'Theme WHIMSICAL'}, 
            inplace = True)


# In[98]:


data = pd.merge(data, style, how = 'left', left_index = True, right_index = True)
data = pd.merge(data, theme, how = 'left', left_index = True, right_index = True)
data.head()


# In[99]:


style_theme_cols = ['Style ADD UP', 'Style BINGO',
       'Style COORDINATE', 'Style CROSSWORD', 'Style EXTENDED PLAY',
       'Style FIND NUMBER', 'Style FIND SYMBOL', 'Style KEY NUMBER MATCH',
       'Style KEY SYMBOL MATCH', 'Style MATCH 2', 'Style MATCH 3',
       'Style MAZE', 'Style MULTIPLE', 'Style OTHER', 'Style SLINGO',
       'Style TIC-TAC-TOE', 'Theme ARIZONA', 'Theme BINGO', 'Theme CARDS',
       'Theme CROSSWORD', 'Theme GAMING', 'Theme LICENSED PROPERTY',
       'Theme LUCK', 'Theme LUCKY NUMBERS', 'Theme MONEY', 'Theme OTHER',
       'Theme SEASONAL', 'Theme SLINGO', 'Theme SPECIALTY', 'Theme SPORTS',
       'Theme WHIMSICAL']

# find average number of times theme/style purchased each year
for col in style_theme_cols:
    data[col] = data[col]/data['Active length']
    
data.head()


# In[100]:


games = pd.pivot_table(data = dff, index = 'Retailer', columns = 'Game Number', 
                       values = 'GGR', aggfunc = np.sum, fill_value = 0)

games.head()


# In[101]:


data = pd.merge(data, games, how = 'left', left_index = True, right_index = True)
data.head()


# In[102]:


game_cols = games.columns

for col in game_cols:
    data[col] = data[col]/data['Active length']

data.head()


# In[103]:


data.shape


# In[104]:


data.to_excel("FinalResults.xlsx")

