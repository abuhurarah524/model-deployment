#!/usr/bin/env python
# coding: utf-8

# ### Importing Libararies

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ### Loading dataset

# In[ ]:


df=pd.read_csv("./LoanExport/LoanExport.csv",low_memory=False)


# ## Data Pre-Processing & EDA

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ### Realizations
# 1. Average credit score is 709
# 2. Average MIP is 9
# 3. Average OCLTV is 77
# 4. Average DTI is 30
# 5. Average OrigUPB is 124940
# 6. Average LTV is 77
# 7. Average Interest rate is 7%

# ### Checking Null Values

# In[ ]:


print(df.isnull().sum().sort_values(ascending=False),"\n\n",df.isnull().sum()/df.shape[0] *100,"\n\n")


# ### Numeric and Categoric Columns

# In[ ]:


#list of all the numeric columns
num = df.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = df.select_dtypes('object').columns.to_list()


# In[ ]:


num


# In[ ]:


cat


# In[ ]:


# check unique values in categorical cols 
[df[category].value_counts() for category in cat[1:]]


# In[ ]:


# check unique values in numerical cols 
[df[numerical].value_counts() for numerical in num[1:]]


# #### There are some columns having value 'X' which means there is no information available. So we are changing it with NaN values.

# In[ ]:


df['FirstTimeHomebuyer'] = df['FirstTimeHomebuyer'].replace('X', np.nan)
df['MSA'] = df['MSA'].replace('X    ',np.nan)
df['PPM'] = df['PPM'].replace('X', np.nan)
df['PropertyType'] = df['PropertyType'].replace('X ', np.nan)
df['NumBorrowers'] = df['NumBorrowers'].replace('X ', np.nan)


# ### Checking Nulls after replacing X values

# In[ ]:


df.isnull().sum()


# #### Handling Missing Values in FirstTimeHomebuyer, PPM and NumBorrowers using mode imputation and dropping MSA and SellerName columns

# In[ ]:


# Mode imputation
df['FirstTimeHomebuyer'].fillna(df['FirstTimeHomebuyer'].mode()[0], inplace=True)
df['PPM'].fillna(df['PPM'].mode()[0], inplace=True)
df['NumBorrowers'].fillna(df['NumBorrowers'].mode()[0], inplace=True)
df['PropertyType'].fillna(df['PropertyType'].mode()[0], inplace=True)
df['MSA'].fillna(df['MSA'].mode()[0], inplace=True)
df['SellerName'].fillna(df['SellerName'].mode()[0], inplace=True)


# In[ ]:


df.isnull().sum()


# ### Converting FirstPaymentDate & MaturityDate into correct date format

# In[ ]:


df['FirstPaymentDate'] = pd.to_datetime(df['FirstPaymentDate'], format='%Y%m')
df['MaturityDate'] = pd.to_datetime(df['MaturityDate'], format='%Y%m')


# In[ ]:


df['FirstPaymentDate'].value_counts()


# In[ ]:


df['MaturityDate'].value_counts()


# #### Creating Pre-Payment Risk(Target) Variable

# In[ ]:


df['PrepaymentRisk'] = df['CreditScore'] * df['DTI']

# Scale the prepayment risk values between 0 and 1
scaler = MinMaxScaler()
df['PrepaymentRisk'] = scaler.fit_transform(df[['PrepaymentRisk']])


# ### Numeric and Categoric Dataframes

# In[ ]:


cat = df.select_dtypes('object').columns.to_list()
num = df.select_dtypes('number').columns.to_list()


# In[ ]:


#numeric df
BM_num =  df[num]
#categoric df
BM_cat = df[cat]


# ## Univariant Analysis

# ### Categoric Columns

# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='FirstTimeHomebuyer' , data=df ,palette='mako')
plt.xlabel('FirstTimeHomebuyer', fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='Occupancy' , data=df ,palette='mako')
plt.xlabel('Occupancy', fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='Channel' , data=df ,palette='mako')
plt.xlabel('Channel', fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='PPM' , data=df ,palette='mako')
plt.xlabel('PPM', fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(27,15))
sns.countplot(x='PropertyState' , data=df ,palette='mako')
plt.xlabel('PropertyState', fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='PropertyType' , data=df ,palette='mako')
plt.xlabel('PropertyType', fontsize=14)
plt.show()


# In[ ]:


plt.figure(figsize=(6,4))
sns.countplot(x='LoanPurpose' , data=df ,palette='mako')
plt.xlabel('LoanPurpose', fontsize=14)
plt.show()


# #### Categoric columns realizations
# 
# * `FirstTimeHomebuyer` - Most of the the borrowers are not first-time homebuyer.
# * `Occupancy` - Most of the properties are `owner-occupied`
# * `Channel` - Origination channel is mostly `Third-party (T)`. 
# * `PPM` - Almost most of the loans borrowed by borrowers have no prepayment penalty. 
# * `PropertyState` - Most of the properties are loacted in '`California' (CA)`.
# * `PropertyType` - Most of the properties are for `Single-Family (SF)`.
# * `LoanPurpose` - The loan purpose for `Purchase(P)` have most values.
# 

# In[ ]:


num


# ### Numeric Columns 

# In[ ]:


cols_num=[
'CreditScore',
 'MIP',
 'Units',
 'OCLTV',
 'DTI',
 'OrigUPB',
 'LTV',
 'OrigInterestRate',
 'OrigLoanTerm',
 'EverDelinquent',
 'MonthsDelinquent',
 'MonthsInRepayment',
 'PrepaymentRisk']


# In[ ]:


for col in cols_num:
    plt.figure(figsize=(8,6))
    sns.histplot(df[col], kde=True, bins=50)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f"Distribution of {col}")
    plt.show()


# ### Realizations
# 1. Credit score follows a left skewed distribution. Most values are concentrated within 500-800 points
# 2. Interest Rates are lying between 6-8%
# 3. OrigUPB values are lying between 100000-300000
# 4. LTV and OCLTV follows a left skewed distribution. Highest value of both is between 80 to 100

# ##  Bivariant Analysis of Numerical Columns

# In[ ]:


# Create a boxplot to compare the credit scores of first-time homebuyers vs. repeat homebuyers
sns.boxplot(x='FirstTimeHomebuyer', y='CreditScore', data=df)
plt.show()


# In[ ]:


# Create a scatter plot to visualize the relationship between debt-to-income ratio and original interest rate
sns.scatterplot(x='DTI', y='OrigInterestRate', data=df)
plt.show()


# In[ ]:


# Stacked bar chart of PropertyType vs PrepaymentRisk
pd.crosstab(df['PropertyType'], df['EverDelinquent']).plot(kind='bar', stacked=True)
plt.show()


# In[ ]:


sns.histplot(data=df, x='DTI', hue='EverDelinquent', element='step')


# In[ ]:


# Plot line graph
df.groupby(['FirstPaymentDate'])['CreditScore'].mean().plot()
plt.xlabel('First Payment Date')
plt.ylabel('Average Credit Score')
plt.show()


# In[ ]:


df.groupby(['FirstPaymentDate'])['OrigInterestRate'].mean().plot()
plt.xlabel('First Payment Date')
plt.ylabel('Average OrigInterestRate')
plt.show()


# In[ ]:


df.groupby(['FirstPaymentDate'])['DTI'].mean().plot()
plt.xlabel('First Payment Date')
plt.ylabel('Average DTI')
plt.show()


# ## Multi-Variant Analysis of Numerical Columns

# In[ ]:


# Create a pairplot to visualize the relationships between numeric variables
sns.pairplot(df[['CreditScore', 'DTI', 'OrigUPB', 'LTV', 'OrigInterestRate']])
plt.show()


# In[ ]:


num


# ## Feature Engineering

# In[ ]:


df['Location'] = df['PostalCode'].astype(str) + '' + df['PropertyState']


# In[ ]:


df.head()


# In[ ]:


df = df.drop(['PostalCode','PropertyState'], axis=1)


# In[ ]:


# Extract month and year from FirstPaymentDate
df['FirstPaymentMonth'] = df['FirstPaymentDate'].dt.month
df['FirstPaymentYear'] = df['FirstPaymentDate'].dt.year

# Extract month and year from MaturityDate
df['MaturityMonth'] = df['MaturityDate'].dt.month
df['MaturityYear'] = df['MaturityDate'].dt.year


# In[ ]:


df = df.drop(['FirstPaymentDate','MaturityDate'], axis=1)


# In[ ]:


df.head()


# ### Finding Outliers 

# In[ ]:


# find and print outliers for each numerical column
for col in num:
    q1 = df[col].quantile(0.15)
    q3 = df[col].quantile(0.85)
    iqr = q3 - q1
    lower_bound = q1 - 1 * iqr
    upper_bound = q3 + 1 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    print(f"{col} outliers:")
    print(outliers)


# In[ ]:


# calculate median for each column
medians = df[num].median()


# ### Handling Outliers

# In[ ]:


# replace outliers with median
for col in num:
    q1 = df[col].quantile(0.15)
    q3 = df[col].quantile(0.85)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[col] = df[col].apply(lambda x: medians[col] if (x < lower_bound or x > upper_bound) else x)


# ### Numerical column after handling outliers

# In[ ]:


# create boxplots for each numerical column after handling outliers
for col in num:
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=col, ax=ax)
    ax.set_title(f"Boxplot of {col}")
    plt.show()


# In[ ]:


#list of all the numeric columns
num = df.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = df.select_dtypes('object').columns.to_list()


# In[ ]:


Num_Cols=df[num]


# In[ ]:


cat_label=[
 'FirstTimeHomebuyer',
 'Occupancy',
 'Channel',
 'PPM',
 'LoanPurpose',
 'NumBorrowers']


# In[ ]:


cat_label=df[cat_label]


# #### Label Encoding

# In[ ]:


cat_label.apply(lambda x: x.nunique()) #checking the number of unique values in each column


# In[ ]:


le = LabelEncoder()
Label = [
 'FirstTimeHomebuyer',
 'Occupancy',
 'Channel',
 'PPM',
 'LoanPurpose',
 'NumBorrowers']

for i in Label:
    df[i] = le.fit_transform(df[i])
    
df.tail()


# In[ ]:


for variable in ['Location','PropertyType','ServicerName']:
    count_map=df[variable].value_counts().to_dict() ## Calculating the number of observations present in each feature
    df[variable]=df[variable].map(count_map) ## Encoding the variables with the count of their observations


# In[ ]:


for var in ['CreditScore','DTI','LTV','OrigUPB','OrigLoanTerm','MonthsInRepayment','PrepaymentRisk']:
    df[var]=df[var].round(3).astype(int) ## rounding off the values to 3 decimal point to simplify the process


# In[ ]:


num = df.select_dtypes('number').columns.to_list()


# In[ ]:


num


# ### Dropping irrelevant features

# In[ ]:


df.drop(['SellerName','ProductType','LoanSeqNum'],axis=1,inplace=True) ## Dropping the irrelevant features


# In[ ]:


df1=df.copy()


# ### Correlation

# In[ ]:


corr=df1.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm', )
plt.show()


# In[ ]:


# Step 2: Correlation analysis
ever_delinquent_corr = corr['EverDelinquent'].abs().sort_values(ascending=False)
# Select variables with correlation above a threshold
significant_corr_vars = ever_delinquent_corr[ever_delinquent_corr > 0.02].index.tolist()


# In[ ]:


significant_corr_vars


# In[ ]:


df1.head()


# In[ ]:


X=df1.drop('EverDelinquent',axis=1) ## independent variables
y=df1['EverDelinquent'] ## dependent variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) ## 20% data is for test and 80% goes to
## training
X_train.shape,X_test.shape


# In[ ]:


scaler=StandardScaler() ## object for standardscaler
X_train_sc=scaler.fit_transform(X_train) ## fitting the standardscaler object in train set
X_test_sc=scaler.transform(X_test) 


# ### PCA

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(n_components=10)


# In[ ]:


## fitting our data into the dimensional space 
X_train_trf=pca.fit_transform(X_train_sc)
X_test_trf=pca.transform(X_test_sc)


# In[ ]:


pca.explained_variance_


# In[ ]:


pca.components_


# In[ ]:


pca.explained_variance_ratio_ 


# In[ ]:


## Finding the optimum number of components
pca=PCA(n_components=None) ## Principal components equal to the number of variables in the data
X_train_trf=pca.fit_transform(X_train_sc)
X_test_trf=pca.transform(X_test_sc)


# In[ ]:


X_train_trf


# In[ ]:



pca.explained_variance_.shape


# In[ ]:


pca.components_.shape 


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:



np.cumsum(pca.explained_variance_ratio_)


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_)) ## plotting the cumulative distribution function
plt.xlabel('No of components')
plt.ylabel('% of variance explained')


# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif
# Step 1: Instantiate the SelectKBest class with the desired score function
k = 15  # Number of top features to select
selector = SelectKBest(score_func=f_classif, k=k)

# Step 2: Apply SelectKBest to the data
X_new = selector.fit_transform(X, y)

# Step 3: Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Step 4: Get the names of the selected features
selected_features = X.columns[selected_feature_indices]

print("Top 15 features:")
print(selected_features)


# In[ ]:


selected_features


# ### Mutual Information

# In[ ]:


# Mutual Information
from sklearn.feature_selection import SelectKBest, mutual_info_classif
mi_selector = SelectKBest(score_func=mutual_info_classif, k=15)  # Select top 15 features
X_mi = mi_selector.fit_transform(X, y)
mi_scores = mi_selector.scores_
mi_features = X.columns[mi_selector.get_support()].tolist()


# In[ ]:


mi_features


# ### Model Building using Top 15 features obtained from PCA

# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.2,random_state=0) ## 20% data is for test and 80% goes to


# ### Gradient Boosting

# In[ ]:


# Train Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Make predictions using Gradient Boosting model
gb_preds = gb_model.predict(X_test)

# Print classification report and accuracy for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Classification Report:")
print(classification_report(y_test, gb_preds))
print("Accuracy:", accuracy_score(y_test, gb_preds)*100)


# In[ ]:


# Print confusion matrix for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gb_preds))
print()


# ### XGBoost

# In[ ]:


# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions using XGBoost model
xgb_preds = xgb_model.predict(X_test)

# Print classification report and accuracy for XGBoost model
print("XGBoost Classifier:")
print("Classification Report:")
print(classification_report(y_test, xgb_preds))
print("Accuracy:", accuracy_score(y_test, xgb_preds)*100)


# In[ ]:


# Print confusion matrix for XGBoost model
print("XGBoost Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))
print()


# ### Model Building using Top 15 features obtained from Mutual Information

# In[ ]:


X_train_mi,X_test_mi,y_train_mi,y_test_mi=train_test_split(X_mi,y,test_size=0.2,random_state=0) ## 20% data is for test and 80% goes to


# ### Gradient Boosting

# In[ ]:


# Train Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_mi, y_train_mi)

# Make predictions using Gradient Boosting model
gb_preds = gb_model.predict(X_test_mi)

# Print classification report and accuracy for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Classification Report:")
print(classification_report(y_test_mi, gb_preds))
print("Accuracy:", accuracy_score(y_test_mi, gb_preds)*100)


# In[ ]:


# Print confusion matrix for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test_mi, gb_preds))
print()


# ### XGBoost

# In[ ]:


# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_mi, y_train_mi)

# Make predictions using XGBoost model
xgb_preds = xgb_model.predict(X_test_mi)

# Print classification report and accuracy for XGBoost model
print("XGBoost Classifier:")
print("Classification Report:")
print(classification_report(y_test_mi, xgb_preds))
print("Accuracy:", accuracy_score(y_test_mi, xgb_preds)*100)


# In[ ]:


# Print confusion matrix for XGBoost model
print("XGBoost Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test_mi, xgb_preds))
print()


# ## Prepayment Risk Prediction

# In[62]:


data=pd.read_csv("./LoanExport/LoanExport.csv",low_memory=False)


# In[63]:


data.head()


# In[64]:


data['PrepaymentRisk'] = data['CreditScore'] * data['DTI']

# Scale the prepayment risk values between 0 and 1
scaler = MinMaxScaler()
data['PrepaymentRisk'] = scaler.fit_transform(data[['PrepaymentRisk']])


# In[65]:


for var in ['MSA','PropertyType','NumBorrowers']: ## Loop containing variables used for handling nan values
    data[var]=data[var].str.strip() ## Removing the white space using str.strip()
    data[var]=data[var].replace('X','Other') ## Replacing nan values with a category "Other"


# In[66]:


for var in ['FirstTimeHomebuyer','SellerName']:
    mode=data[var].mode()
    data[var]=data[var].fillna(mode)[0]


# In[67]:


data['PPM']=data['PPM'].replace('X','Other') ## Replacing nan values with "Other"
data['PostalCode']=data['PostalCode'].str.strip()
data['PostalCode']=data['PostalCode'].replace('X',np.nan)
data['PostalCode']=data['PostalCode'].fillna(mode)[0]


# In[68]:


data.isnull().sum()


# In[69]:


# Function for capping outliers with 85th percentile and 15th percentile
def cap_outliers(data,variable):
    
    upper_limit=data[variable].quantile(0.85)
    lower_limit=data[variable].quantile(0.15)
    data[variable]=np.where(data[variable]>=upper_limit,upper_limit,
                         np.where(data[variable]<=lower_limit,lower_limit,
                                 data[variable])) ## We want values larger than upper limit and lower than lower limit


# In[70]:


variables=['CreditScore','FirstPaymentDate','MaturityDate','LTV','OCLTV','OrigUPB','OrigInterestRate','OrigLoanTerm',
          'MonthsDelinquent','MonthsInRepayment']

for variable in variables:
    cap_outliers(data,variable) ## Loop to cap outliers for the variables in the list


# In[71]:


## Function to plot distribution after removing outliers
def distribution_plot(data,variable):
    plt.figure(figsize=(8,6))
    
    plt.subplot(1,2,1)
    data.boxplot(column=variable)
    plt.title('Boxplot after handling outliers')
    
    plt.subplot(1,2,2)
    sns.distplot(data[variable])
    plt.title('Distribution after handling outliers')
    
    plt.show()


# In[72]:


le = LabelEncoder()
Label = [
 'FirstTimeHomebuyer',
 'Occupancy',
 'Channel',
 'PPM',
 'LoanPurpose',
 'NumBorrowers',
 'PropertyType',
 'PropertyState'
 ]

for i in Label:
    data[i] = le.fit_transform(data[i])
    
data.tail()


# In[73]:


for var in ['ServicerName','MSA']:
    frequency_map=(data[var].value_counts()/len(data)).to_dict() ## Calculating the frequency of the variables and convert them into
    ## a dictionary
    data[var]=data[var].map(frequency_map) ## Encoding with the frequency of the variables through loop


# In[74]:


data.drop(['SellerName','ProductType','LoanSeqNum'],axis=1,inplace=True) ## Dropping the irrelevant features


# In[75]:


for var in ['CreditScore','FirstPaymentDate','MaturityDate','LTV','OCLTV','OrigUPB','OrigLoanTerm','MonthsInRepayment']:
    data[var]=data[var].round(1).astype(int) ## rounding off the values to 1 decimal point to simplify the process


# In[76]:


data['FirstPaymentDate']=data['FirstPaymentDate'].astype(str) ## converting the feature into string
data['FirstPaymentDate']=pd.to_datetime(data['FirstPaymentDate'],format="%Y%m") ## converting to appropriate date-time format 
## using year and month

data['MaturityDate']=data['MaturityDate'].astype(str)
data['MaturityDate']=pd.to_datetime(data['MaturityDate'],format="%Y%m")


# In[77]:


data['FirstPaymentYear']=data['FirstPaymentDate'].dt.year ## extracting year from firstpaymentdate
data['FirstPaymentMonth']=data['FirstPaymentDate'].dt.month ## extracting month from firstpaymentdate
data['MaturityYear']=data['MaturityDate'].dt.year ## extracting year from maturitydate
data['MaturityMonth']=data['MaturityDate'].dt.month ## extracting month from maturitydate


# In[78]:


data.drop(['FirstPaymentDate','MaturityDate'],axis=1,inplace=True)


# In[79]:


data.head()


# In[ ]:


data.to_csv('result.csv')


# In[80]:


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

X=data.drop('PrepaymentRisk',axis=1) ## independent variables
y=data['PrepaymentRisk'] ## dependent variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) ## 20% data is for test and 80% goes to
## training
X_train.shape,X_test.shape


# In[81]:


scaler=StandardScaler() ## object for standardscaler
X_train=scaler.fit_transform(X_train) ## fitting the standardscaler object in train set
X_test=scaler.transform(X_test) ## fitting the scaler object in test set


# In[82]:


## Finding the optimum number of components
pca=PCA(n_components=None) ## Principal components equal to the number of variables in the data
X_train_trf=pca.fit_transform(X_train)
X_test_trf=pca.transform(X_test)


# In[84]:


np.cumsum(pca.explained_variance_ratio_) ## cumulative percentages of eigenvectors


# In[85]:


plt.plot(np.cumsum(pca.explained_variance_ratio_)) ## plotting the cumulative distribution function
plt.xlabel('No of components')
plt.ylabel('% of variance explained')


# In[86]:


from sklearn.feature_selection import f_regression ## F-score
ordered_rank_features=SelectKBest(score_func=f_regression,k=15) ## This will give the top 15 features according to their F-score
ordered_feature=ordered_rank_features.fit(X,y) ## Fitting the algorithm to the dataset. Higher the F-score, more relevant
## is the feature with the target


# In[87]:


ordered_feature.scores_ ## p-values corresponding to each F-score. Lower the p-value, more relevant the feature is with the
## target


# In[88]:


datascores=pd.DataFrame(ordered_feature.scores_) ## converting the F-scores numpy array to a dataframe
datacolumns=pd.DataFrame(X.columns) ## Converting the independent features array to a dataframe


# In[89]:


features_rank=pd.concat([datacolumns,datascores],axis=1) ## merging the two dataframes
features_rank.columns=['Features','Scores'] ## naming the columns
features_rank


# In[90]:


features_rank.nlargest(15,'Scores') ## We need only 15 features to explain the maximum amount of variance in the data


# In[91]:


X=data[['DTI','MaturityMonth','FirstPaymentMonth','OCLTV','OrigInterestRate','LTV','MIP','NumBorrowers',
     'OrigUPB','PropertyState','LoanPurpose','Channel','PostalCode','PropertyType','CreditScore']]
y=data['PrepaymentRisk']


# In[92]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape


# In[93]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ### Model building using PCA and top features

# In[94]:


# Train the linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Make predictions on the test set
linear_reg_preds = linear_reg.predict(X_test)

# Evaluate linear regression model
linear_reg_mse = mean_squared_error(y_test, linear_reg_preds)
linear_reg_mae = mean_absolute_error(y_test, linear_reg_preds)
linear_reg_r2 = r2_score(y_test, linear_reg_preds)

print("Linear Regression Metrics:")
print("Mean Squared Error (MSE):", linear_reg_mse)
print("Mean Absolute Error (MAE):", linear_reg_mae)
print("R-squared (R2):", linear_reg_r2)
print('Training Accuracy:',linear_reg.score(X_train,y_train))
print('Test Accuracy:',np.round(linear_reg.score(X_test,y_test),2))


# In[95]:


filename="Linear_PCA_model.pkl"


# In[96]:


import pickle
pickle.dump(linear_reg,open(filename,'wb'))


# In[97]:


# Train the Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Make predictions on the test set
lasso_preds = lasso.predict(X_test)

# Evaluate Lasso model
lasso_mse = mean_squared_error(y_test, lasso_preds)
lasso_mae = mean_absolute_error(y_test, lasso_preds)
lasso_r2 = r2_score(y_test, lasso_preds)

print("\nLasso Metrics:")
print("Mean Squared Error (MSE):", lasso_mse)
print("Mean Absolute Error (MAE):", lasso_mae)
print("R-squared (R2):", lasso_r2)
print('Training Accuracy:',lasso.score(X_train,y_train))
print('Test Accuracy:',np.round(lasso.score(X_test,y_test),2))


# In[98]:


filename="Lasso_PCA_model.pkl"


# In[99]:


pickle.dump(linear_reg,open(filename,'wb'))


# ### Model building using MI and top features

# In[100]:


from sklearn.feature_selection import mutual_info_regression
mi_selector = SelectKBest(score_func=mutual_info_regression, k=15)  # Select top 15 features
X_mi = mi_selector.fit_transform(X, y)
mi_scores = mi_selector.scores_
mi_features = X.columns[mi_selector.get_support()].tolist()


# In[101]:


mi_features


# In[102]:


X_train,X_test,y_train,y_test=train_test_split(X_mi,y,test_size=0.3,random_state=0)


# ### LinearRegression

# In[103]:


lin=LinearRegression()
lin.fit(X_train,y_train)
y_pred=lin.predict(X_test)
print('Linear Regression Metrics:')
print('R2 Score:',r2_score(y_pred,y_test))
print('Mean Squared Error:',mean_squared_error(y_pred,y_test))
print('Mean Absolute Error:',mean_absolute_error(y_pred,y_test))


# In[104]:


print('Training Accuracy:',lin.score(X_train,y_train))
print('Test Accuracy:',np.round(lin.score(X_test,y_test),2))


# In[105]:


filename="Linear_MI_model.pkl"
pickle.dump(lin,open(filename,'wb'))


# ### LassoRegression

# In[106]:


# Train the Lasso model
lasso_mi = Lasso(alpha=0.1)
lasso_mi.fit(X_train, y_train)

# Make predictions on the test set
lasso_preds = lasso_mi.predict(X_test)

# Evaluate Lasso model
lasso_mse = mean_squared_error(y_test, lasso_preds)
lasso_mae = mean_absolute_error(y_test, lasso_preds)
lasso_r2 = r2_score(y_test, lasso_preds)

print("\nLasso Metrics:")
print("Mean Squared Error (MSE):", lasso_mse)
print("Mean Absolute Error (MAE):", lasso_mae)
print("R-squared (R2):", lasso_r2)
print('Training Accuracy:',lasso.score(X_train,y_train))
print('Test Accuracy:',np.round(lasso.score(X_test,y_test),2))


# In[107]:


filename="Lasso_MI_model.pkl"
pickle.dump(lasso_mi,open(filename,'wb'))


# ## Classification

# ### Model building using PCA and top features

# In[108]:


X=data[['DTI','MaturityMonth','FirstPaymentMonth','OCLTV','OrigInterestRate','LTV','MIP','NumBorrowers',
     'OrigUPB','PropertyState','LoanPurpose','Channel','PostalCode','PropertyType','CreditScore']]
y=data['EverDelinquent']


# In[109]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) ## 20% data is for test and 80% goes to


# In[110]:


# Train Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Make predictions using Gradient Boosting model
gb_preds = gb_model.predict(X_test)

# Print classification report and accuracy for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Classification Report:")
print(classification_report(y_test, gb_preds))
print("Accuracy:", np.round(accuracy_score(y_test, gb_preds)*100,2))


# In[111]:


# Print confusion matrix for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, gb_preds))
print()


# In[112]:


filename="GB_PCA_model.pkl"
pickle.dump(gb_model,open(filename,'wb'))


# In[113]:


# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions using XGBoost model
xgb_preds = xgb_model.predict(X_test)

# Print classification report and accuracy for XGBoost model
print("XGBoost Classifier:")
print("Classification Report:")
print(classification_report(y_test, xgb_preds))
print("Accuracy:", np.round(accuracy_score(y_test, xgb_preds)*100,2))


# In[ ]:


# Print confusion matrix for XGBoost model
print("XGBoost Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))
print()


# In[ ]:


filename="XGB_PCA_model.pkl"
pickle.dump(xgb_model,open(filename,'wb'))


# ### Model Building using Top 15 features obtained from Mutual Information

# In[114]:


X_train_mi,X_test_mi,y_train_mi,y_test_mi=train_test_split(X_mi,y,test_size=0.2,random_state=0) ## 20% data is for test and 80% goes to


# ### Gradient Boosting

# In[116]:


# Train Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_mi, y_train_mi)

# Make predictions using Gradient Boosting model
gb_preds = gb_model.predict(X_test_mi)

# Print classification report and accuracy for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Classification Report:")
print(classification_report(y_test_mi, gb_preds))
print("Accuracy:",np.round( accuracy_score(y_test_mi, gb_preds)*100,2))


# In[117]:


# Print confusion matrix for Gradient Boosting model
print("Gradient Boosting Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test_mi, gb_preds))
print()


# ### XGBoost

# In[118]:


# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train_mi, y_train_mi)

# Make predictions using XGBoost model
xgb_preds = xgb_model.predict(X_test_mi)

# Print classification report and accuracy for XGBoost model
print("XGBoost Classifier:")
print("Classification Report:")
print(classification_report(y_test_mi, xgb_preds))
print("Accuracy:", np.round(accuracy_score(y_test_mi, xgb_preds)*100,2))


# In[119]:


# Print confusion matrix for XGBoost model
print("XGBoost Classifier:")
print("Confusion Matrix:")
print(confusion_matrix(y_test_mi, xgb_preds))
print()

