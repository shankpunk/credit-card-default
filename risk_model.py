#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Prediction of Credit Card Default

### Tutorial for beginners in classification analysis using scikit-learn.


# This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.
# 
# 
# # Table of Content
# 
# * [Objectives](#obj)
# * [Importing packages and loading data](#imp)
# * [Feature Engineering](#fe)
# * [Exploratory Data Analysis (EDA)](#eda)
#     * [Mapping the target: categorizing](#map)
#     * [Descriptive Statistics](#stat)
#     * [Standardizing and plotting features](#std)
#     * [Correlation](#corr)
# * [Machine Learning: Classification models](#ml)
#     * [Feature Selection](#fs)
#     * [Spliting the data: train and test](#sp)
#     * [Logistic Regression (original data)](#lr1)
#     * [Logistic Regression (standardized features)](#lr2)
#     * [Logistic Regression (most important features)](#lr3)
#     * [ExtraTree-decision](#tree)
#     * [Random-Forest Classifier](#rf)
# * [Comparison of model performance](#sum)
#     * [Receiver operating characteristic (ROC) Curve](#roc)
#     * [Mean Accuracy (coss-validation)](#ac)
#     * [Precision, Recall, F1-score](#m)

# <a id='obj'></a>
# ## Objectives:<br>
# -       Identify the key drivers that determine the likelihood of credit card default.
# -       Predict the likelihood of credit card default for customers of the Bank.

# <a id='imp'></a>
# ## Importing packages and loading data

# In[26]:


# here we will import the libraries used for machine learning
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O, data manipulation
import matplotlib.pyplot as plt # this is used for the plot the graph
import seaborn as sns # used for plot interactive graph.
from pandas import set_option
plt.style.use('ggplot') # nice plots

from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.linear_model import LogisticRegression # to apply the Logistic regression
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold # for cross validation
from sklearn.model_selection import GridSearchCV # for tuning parameter
from sklearn.model_selection import RandomizedSearchCV  # Randomized search on hyper parameters.
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import imblearn


import os

#print(os.listdir("../"))


# In[27]:

def process_data(input_file, delete_nan=False):

    data = pd.read_csv(input_file, skipinitialspace=True)
    data.sample(5)

    #Print unique Occupation types
    #print(data['occupation_type'].unique())


    # There are variables that need to be converted to categories:
    # * __SEX:__ Gender   
    #                     1 = M 
    #                     2 = F
    #                     3 = OTHER
    # * OWNS_CAR:__     
    #                      1 = Y 
    #                      2 = N 
    #                      3 = OTHER
    # * OWNS_HOUSE:__  
    #                     1 = Y
    #                     2 = N
    #                     3 = OTHER
    #                  
    # * OCCUPATION_TYPE
    #                  'Unknown' 'Laborers' 'Core staff' 'Accountants' 'High skill tech staff'
    #                  'Sales staff' 'Managers' 'Drivers' 'Medicine staff' 'Cleaning staff'
    #                  'HR staff' 'Security staff' 'Cooking staff' 'Waiters/barmen staff'
    #                  'Low-skill Laborers' 'Private service staff' 'Secretaries'
    #                  'Realty agents' 'IT staff'

    # <a id='fe'></a>
    # ## Feature engineering
    # Try a few different things to check accuracy. Like
    # 1) removing incomplete data
    # 2) Replacing incomplete data with mean()
    # 3) Increasing the number of defaults

    if delete_nan:
        data.dropna(inplace=True)

    data.rename(columns={"credit_card_default": "Default"}, inplace=True)
    data.rename(columns={"credit_limit_used(%)": "credit_limit_used"}, inplace=True)
    data.drop('name', axis = 1, inplace =True) # drop column "name"
    customer_id = data['customer_id']
    data.drop('customer_id', axis = 1, inplace =True) # drop column "name"
    #data.drop('name', axis = 1, inplace =True) # drop column "name"
    data.info()


    # In[29]:


    # Separating features and target
    #y = data.Default     # target default=1 or non-default=0


    # In[30]:


    data[['gender', 'owns_car', 'owns_house', 'occupation_type']] = data[['gender', 'owns_car', 'owns_house', 'occupation_type']].astype('str')

    data['gender'].fillna(3, inplace = True)

    data['gender'].replace(to_replace=['M', 'F', 'nan', 'XNA'], value=[1, 0, 2, 2], inplace = True)

    data['gender'].unique()


    #
    # replace String in cars and house owned

    # In[31]:



    data['owns_car'].fillna(3, inplace = True)
    data['owns_car'].replace(to_replace=['Y', 'N', 'nan'], value=[1, 0, 2], inplace = True)
    data['owns_car'].unique()


    # After grouping, the education column has the following categories:

    # In[32]:


    data['owns_house'].fillna(0, inplace = True)
    data['owns_house'].replace(to_replace=['Y', 'N', 'nan'], value=[1, 0, 2], inplace = True)
    data['owns_house'].unique()

    occupation = [i for i in range(19)]

    data['occupation_type'].fillna(0, inplace = True)
    data['occupation_type'].replace(to_replace=['Unknown', 'Laborers', 'Core staff', 'Accountants', 'High skill tech staff',
                        'Sales staff', 'Managers', 'Drivers', 'Medicine staff', 'Cleaning staff', 'HR staff',
                        'Security staff', 'Cooking staff', 'Waiters/barmen staff', 'Low-skill Laborers', 'Private service staff',
                        'Secretaries', 'Realty agents', 'IT staff'], value=occupation, inplace = True)
    data['occupation_type'].unique()


    #

    # Similarly, the column 'marriage' should have three categories: 1 = married, 2 = single, 3 = others but it contains a category '0' which will be joined to the category '3'.

    # In[33]:



    data['migrant_worker'].unique()
    data['migrant_worker'] = data['migrant_worker'].fillna(2)
    data['migrant_worker'].unique()

    data['no_of_children'].unique()
    children_mean = data['no_of_children'].mean()


    # Fill missing value with mean.

    data['no_of_children'] = data['no_of_children'].fillna(round(children_mean))
    data['no_of_days_employed'] = data['no_of_days_employed'].fillna(round(data['no_of_days_employed'].mean()))
    data['total_family_members'] = data['total_family_members'].fillna(round(data['total_family_members'].mean()))
    data['migrant_worker'] = data['migrant_worker'].fillna(round(data['migrant_worker'].mean()))
    data['yearly_debt_payments'] = data['yearly_debt_payments'].fillna(data['yearly_debt_payments'].mean())
    data['credit_score'] = data['credit_score'].fillna(data['credit_score'].mean())


    return data, customer_id


#train data
data, c_id = process_data('../dataset/train.csv', delete_nan=False)

y = data.Default     # target default=1 or non-default=0
features = data.drop('Default', axis=1, inplace=False)

#test data
data_test, customer_id = process_data('../dataset/test.csv')



# <a id='eda'></a>
# ## Exploratory Data Analysis (EDA)
# 
# <a id='map'></a>
# ### Mapping the target: categorizing
#
# In[35]:


# The frequency of defaults
yes = data.Default.sum()
no = len(data)-yes

# Percentage
yes_perc = round(yes/len(data)*100, 1)
no_perc = round(no/len(data)*100, 1)

import sys
plt.figure(figsize=(7,4))
sns.set_context('notebook', font_scale=1.2)
sns.countplot('Default',data=data, palette="Blues")
plt.annotate('Non-default: {}'.format(no), xy=(-0.3, 15000), xytext=(-0.3, 3000), size=12)
plt.annotate('Default: {}'.format(yes), xy=(0.7, 15000), xytext=(0.7, 3000), size=12)
plt.annotate(str(no_perc)+" %", xy=(-0.3, 15000), xytext=(-0.1, 8000), size=12)
plt.annotate(str(yes_perc)+" %", xy=(0.7, 15000), xytext=(0.9, 8000), size=12)
plt.title('COUNT OF CREDIT CARDS', size=14)
#Removing the frame
plt.box(False);


# <a id='stat'></a>
# ### Descriptive Statistics
# The table below shows the descriptive statistics of the variables of this dataset.

# In[36]:


set_option('display.width', 100)
set_option('precision', 2)

print("SUMMARY STATISTICS OF NUMERIC COLUMNS")
print()
#print(data.describe().T)
data.sample(10)


# In[37]:


# Creating a new dataframe with categorical variables
subset = data[['default_in_last_6months', 'gender', 'owns_car', 'owns_house', 'no_of_children', 'total_family_members', 'migrant_worker',
               'occupation_type', 'prev_defaults', 'Default']]

f, axes = plt.subplots(3, 3, figsize=(20, 15), facecolor='white')
f.suptitle('FREQUENCY OF CATEGORICAL VARIABLES (BY TARGET)')
ax1 = sns.countplot(x="default_in_last_6months", hue="Default", data=subset, palette="Blues", ax=axes[0,0])
ax2 = sns.countplot(x="gender", hue="Default", data=subset, palette="Blues",ax=axes[0,1])
ax3 = sns.countplot(x="owns_car", hue="Default", data=subset, palette="Blues",ax=axes[0,2])
ax4 = sns.countplot(x="owns_house", hue="Default", data=subset, palette="Blues", ax=axes[1,0])
ax5 = sns.countplot(x="no_of_children", hue="Default", data=subset, palette="Blues", ax=axes[1,1])
ax6 = sns.countplot(x="total_family_members", hue="Default", data=subset, palette="Blues", ax=axes[1,2])
ax7 = sns.countplot(x="migrant_worker", hue="Default", data=subset, palette="Blues", ax=axes[2,0])
ax8 = sns.countplot(x="occupation_type", hue="Default", data=subset, palette="Blues", ax=axes[2,1])
ax9 = sns.countplot(x="prev_defaults", hue="Default", data=subset, palette="Blues", ax=axes[2,2]);
#ax10 = sns.countplot(x="default_in_last_6months", hue="Default", data=subset, palette="Blues", ax=axes[2,2]);


# In[38]:


x1 = list(data[data['Default'] == 1]['credit_limit'])
x2 = list(data[data['Default'] == 0]['credit_limit'])
x1[:5], x2[:5]


# In[39]:


plt.figure(figsize=(12,4))
sns.set_context('notebook', font_scale=1.2)
#sns.set_color_codes("pastel")
plt.hist([x1, x2], bins = 40, density=False, color=['steelblue', 'lightblue'])
plt.xlim([0,4000000])
plt.legend(['Yes', 'No'], title = 'Default', loc='upper right', facecolor='white')
plt.xlabel('Limit Balance')
plt.ylabel('Frequency')
plt.title('LIMIT BALANCE HISTOGRAM BY TYPE OF CREDIT CARD')
plt.box(False)
plt.savefig('ImageName', format='png', dpi=200, transparent=True);

## data are distributed in a wide range (below), need to be normalizded.
plt.figure(figsize=(15,3))
ax= data.drop('Default', axis=1).boxplot(data.columns.name, rot=90)
outliers = dict(markerfacecolor='b', marker='p')
ax= features.boxplot(features.columns.name, rot=90, flierprops=outliers)
plt.xticks(size=12)
ax.set_ylim([-5000,100000])
plt.box(False);


#

# Standardization of data was performed; i.e, all features are centered around zero and have variance one. Features were plotted again, using a violin plot.

# In[43]:


stdX = (features - features.mean()) / (features.std())              # standardization
# <a id='corr'></a>
# ### Correlation
# A correlation matrix of all variables is shown in the heatmap below. The only feature with a notable positive correlation with the dependent variable ‘Default’ is re-payment status during the last month (September). The highest negative correlation with default occurs with Limit_Balance, indicating that customers with lower limit balance are more likely to default. It can also be observed that some variables are highly correlated to each other, that is the case of the amount of bill statement and the repayment status in different months.
# 
# Looking at correlations matrix, defined via Pearson function. 

# In[46]:


#  looking at correlations matrix, defined via Pearson function  
corr = data.corr() # .corr is used to find corelation
f,ax = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, cbar = True,  square = True, annot = False, fmt= '.1f',
            xticklabels= True, yticklabels= True
            ,cmap="coolwarm", linewidths=.5, ax=ax)
plt.title('CORRELATION MATRIX - HEATMAP', size=18);


# The heatmat shows that features are correlated with each other (collinearity)
# 
# Uncorrelated data are poentially more useful: discriminatory!
# 
# **What do correlations mean?**<br>
# 
# Lets separately fit correlated and uncorrelated data via linear regression: 

# In[53]:


sns.lmplot(x='credit_limit', y= 'prev_defaults', data = data, hue ='Default',
           palette='coolwarm')
plt.title('Linear Regression: distinguishing between Default and Non-default', size=16)

sns.lmplot(x='credit_score', y= 'prev_defaults', data = data, hue ='Default',
           palette='coolwarm')
plt.title('Linear Regression: Cannot distinguishing between Default and Non-default', size=16)


sns.lmplot(x='prev_defaults', y= 'default_in_last_6months', data = data, hue ='Default',
           palette='coolwarm')
plt.title('Linear Regression: Cannot distinguish between Default and Non-default', size=16);

print('Uncorrelated data are poentially more useful: discrimentory!')


# <a id='ml'></a>
# ## Machine Learning: Classification models
# 
# The classification models used for this analysis are: Logistic Regression, Decision Tree and Random Forest Classifier.<br>
# 
# To build machine learning models the original data was divided into features (X) and target (y) and then split into train (80%) and test (20%) sets. Thus, the algorithms would be trained on one set of data and tested out on a completely different set of data (not seen before by the algorithm).
# 
# <a id='sp'></a>
# ### Splitting the data into train and test sets

# In[54]:


# Original dataset
oversample = SMOTE(sampling_strategy=0.3)

#oversample = ADASYN(sampling_strategy=0.4)

under = RandomUnderSampler(sampling_strategy=0.3)

y = data['Default']
X = data.drop('Default', axis=1)

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, stratify=y, random_state=42)

X.info()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, stratify=y, random_state=42)

#X, y = under.fit_resample(X, y)

X, y = oversample.fit_resample(X, y)

X2, y2 = oversample.fit_resample(X_train, y_train)



# In[55]:


# Dataset with standardized features
#Xstd_train, Xstd_test, ystd_train, ystd_test = train_test_split(stdX,y, test_size=0.20, stratify=y, random_state=42)


# <a id='fs'></a>
# ### Feature Selection
# 
# #### Recursive Feature Elimination
# Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features. This process is applied until all features in the dataset are exhausted. The goal of RFE is to select features by recursively considering smaller and smaller sets of features.

# In[57]:


NUM_FEATURES = 3

# Create the random grid
param_dist = {'n_estimators': [50,100,150,200,250,300,500,750],
               "max_features": [1,2,3,4,5,6,7,8,9,10,14],
               'max_depth': [1,2,3,4,5,6,7,8,9,10, 12, 14, 18]}

rf = RandomForestClassifier()

rf_cv = RandomizedSearchCV(rf, param_distributions = param_dist,
                           cv=3, random_state=0, n_jobs=-1)

rf_cv.fit(X, y)

print("Tuned Random Forest Parameters: %s" % (rf_cv.best_params_))


Ran = RandomForestClassifier(max_depth=8,
                                     max_features=5, n_estimators= 200,
                                     random_state=0)

Ran_2 = RandomForestClassifier(max_depth=8,
                                     max_features=5, n_estimators= 200,
                                     random_state=0)


Ran.fit(X2, y2)
Ran_2.fit(X, y)

main_prediction = Ran_2.predict(data_test)

data_predicted = pd.DataFrame()

data_predicted['customer_id'] = customer_id
data_predicted['credit_card_default'] = main_prediction

print("TOTAL NUMBER OF Defaults: ", main_prediction.astype(bool).sum())
#print("TOTAL NUMBER OF Non-Defaults: ", main_prediction.count - main_prediction.astype(bool).sum())


import time
timestr = time.strftime("%Y%m%d-%H%M%S")

data_predicted.to_csv('Predictions/Credit_Default_'+timestr+'.csv')

y_pred = Ran.predict(X_test)

print('Accuracy:', metrics.accuracy_score(y_pred,y_test))

## 5-fold cross-validation 
cv_scores =cross_val_score(Ran, X, y, cv=5)

# Print the 5-fold cross-validation scores
print()
print(classification_report(y_test, y_pred))
print()
print("Average 5-Fold CV Score: {}".format(round(np.mean(cv_scores),4)),
      ", Standard deviation: {}".format(round(np.std(cv_scores),4)))

plt.figure(figsize=(4,3))
ConfMatrix = confusion_matrix(y_test,Ran.predict(X_test))
sns.heatmap(ConfMatrix,annot=True, cmap="Blues", fmt="d",
            xticklabels = ['Non-default', 'Default'],
            yticklabels = ['Non-default', 'Default'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix - Random Forest");

