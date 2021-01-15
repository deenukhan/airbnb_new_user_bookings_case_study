# Airbnb New User Bookings 
Where will a new guest book their first travel experience?

## About Data

<p>In this challenge, you are given a list of users along with their demographics, web session records, and some summary statistics. You are asked to predict which country a new user's first booking destination will be.&nbsp;All the users in this dataset are from&nbsp;the USA.</p>

<p>There are 12 possible outcomes of the destination country:&nbsp;'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and&nbsp;'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.</p>

<p>The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after <strong>7/1/2014 (note: this is updated on 12/5/15 when the competition restarted)</strong>.&nbsp;In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010.&nbsp;</p>

<h3>Data Files descriptions</h3>
<ul>
<li><strong>train_users.csv</strong>&nbsp;- the training set of users</li>
<li><strong>test_users.csv</strong> - the test set of users</li>
<ul>
<li>id: user id</li>
<li>date_account_created: the date of account creation</li>
<li>timestamp_first_active: timestamp of the first activity, note that it can be earlier than&nbsp;date_account_created or&nbsp;date_first_booking because a user can search before signing up</li>
<li>date_first_booking: date of first booking</li>
<li>gender</li>
<li>age</li>
<li>signup_method</li>
<li>signup_flow:&nbsp;the page a user came to signup up from</li>
<li>language: international language preference</li>
<li>affiliate_channel:&nbsp;what kind of paid marketing</li>
<li>affiliate_provider:&nbsp;where the marketing is e.g. google, craigslist, other</li>
<li>first_affiliate_tracked:&nbsp;whats the first marketing the user interacted with before the signing up</li>
<li>signup_app</li>
<li>first_device_type</li>
<li>first_browser</li>
<li>country_destination: this is the <strong>target variable</strong> you are to predict</li>
</ul>
<li><strong>sessions.csv</strong> -&nbsp;web sessions log for users</li>
<ul>
<li>user_id: to be joined with the column 'id' in users table</li>
<li>action</li>
<li>action_type</li>
<li>action_detail</li>
<li>device_type</li>
<li>secs_elapsed</li>
</ul>
<li><strong>countries.csv&nbsp;</strong>- summary statistics of destination countries in this dataset and their locations</li>
<li><strong>age_gender_bkts.csv</strong> - summary statistics of users' age group, gender, country of destination</li>
<li><strong>sample_submission.csv</strong> -&nbsp;correct format for submitting your predictions</li>
</ul>

## Extracting Data


```python
#Below code is just to copy all the files into current session's drive and creating and deleting few required folders

import os
import zipfile
from tqdm import tqdm

!mkdir airbnb_data
!mkdir temp
!cp "/content/drive/My Drive/Study/Case Study 1/airbnb_data/airbnb_data.zip" /content/

#-q is oppsoite of verbose, -d for decompressing to directory
!unzip -q /content/airbnb_data.zip -d /content/temp/

for zip_files in tqdm(os.listdir('/content/temp')):
    path = os.path.join("/content/temp", zip_files)
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall("/content/airbnb_data")
    os.remove(path)

os.remove("/content/airbnb_data.zip")
os.rmdir("/content/temp")
```

    100%|██████████| 6/6 [00:03<00:00,  1.77it/s]
    

## Reading the Data


```python
#Importing Libraries
import os
import pickle
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.impute import SimpleImputer

#Base Learners
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline
```


```python
#Reading the data
age_gender = pd.read_csv('/content/airbnb_data/age_gender_bkts.csv')
countries = pd.read_csv('/content/airbnb_data/countries.csv')
sessions = pd.read_csv('/content/airbnb_data/sessions.csv')
train_users = pd.read_csv('/content/airbnb_data/train_users_2.csv')
test_users = pd.read_csv('/content/airbnb_data/test_users.csv')
```

## Exploratory Data Analysis

**Most important thing to keep in mind is that, all of our EDA should be alligned to our Target variable, means what is affecting our Target Variable**


```python
#First Let's Checkout the Shape of our datasets
print("Shape of Training data   : ", train_users.shape)
print("Shape of Testing data    : ", test_users.shape)
print("Shape of Countries data  : ", countries.shape)
print("Shape of AgeGender data  : ", age_gender.shape)
print("Shape of Sessions data   : ", sessions.shape)
```

    Shape of Training data   :  (213451, 16)
    Shape of Testing data    :  (62096, 15)
    Shape of Countries data  :  (10, 7)
    Shape of AgeGender data  :  (420, 5)
    Shape of Sessions data   :  (10567737, 6)
    


```python
#Let's check out some basic inofrmation about the data
train_users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 213451 entries, 0 to 213450
    Data columns (total 16 columns):
     #   Column                   Non-Null Count   Dtype  
    ---  ------                   --------------   -----  
     0   id                       213451 non-null  object 
     1   date_account_created     213451 non-null  object 
     2   timestamp_first_active   213451 non-null  int64  
     3   date_first_booking       88908 non-null   object 
     4   gender                   213451 non-null  object 
     5   age                      125461 non-null  float64
     6   signup_method            213451 non-null  object 
     7   signup_flow              213451 non-null  int64  
     8   language                 213451 non-null  object 
     9   affiliate_channel        213451 non-null  object 
     10  affiliate_provider       213451 non-null  object 
     11  first_affiliate_tracked  207386 non-null  object 
     12  signup_app               213451 non-null  object 
     13  first_device_type        213451 non-null  object 
     14  first_browser            213451 non-null  object 
     15  country_destination      213451 non-null  object 
    dtypes: float64(1), int64(2), object(13)
    memory usage: 26.1+ MB
    

1. **We have 213451 Entries with 16 columns, 15 being indepnedant variables and "country_destination" being dependent variable**


```python
test_users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 62096 entries, 0 to 62095
    Data columns (total 15 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       62096 non-null  object 
     1   date_account_created     62096 non-null  object 
     2   timestamp_first_active   62096 non-null  int64  
     3   date_first_booking       0 non-null      float64
     4   gender                   62096 non-null  object 
     5   age                      33220 non-null  float64
     6   signup_method            62096 non-null  object 
     7   signup_flow              62096 non-null  int64  
     8   language                 62096 non-null  object 
     9   affiliate_channel        62096 non-null  object 
     10  affiliate_provider       62096 non-null  object 
     11  first_affiliate_tracked  62076 non-null  object 
     12  signup_app               62096 non-null  object 
     13  first_device_type        62096 non-null  object 
     14  first_browser            62096 non-null  object 
    dtypes: float64(2), int64(2), object(11)
    memory usage: 7.1+ MB
    

1. **In Test data we can see we have 15 columns, obviosuly we don't have our target variable here as we have to predict that.**
2. **one more thing we can notice here is that, date_first_booking column is also given here which don't have any values and also doesn't make any sense to be in testing data, as user haven't booked the destination yet.**
3. **So we will be removing "date_first_booking" column from our training and testing data.**



```python
#Let's checkout how data looks like
train_users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date_account_created</th>
      <th>timestamp_first_active</th>
      <th>date_first_booking</th>
      <th>gender</th>
      <th>age</th>
      <th>signup_method</th>
      <th>signup_flow</th>
      <th>language</th>
      <th>affiliate_channel</th>
      <th>affiliate_provider</th>
      <th>first_affiliate_tracked</th>
      <th>signup_app</th>
      <th>first_device_type</th>
      <th>first_browser</th>
      <th>country_destination</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gxn3p5htnn</td>
      <td>2010-06-28</td>
      <td>20090319043255</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>facebook</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Chrome</td>
      <td>NDF</td>
    </tr>
    <tr>
      <th>1</th>
      <td>820tgsjxq7</td>
      <td>2011-05-25</td>
      <td>20090523174809</td>
      <td>NaN</td>
      <td>MALE</td>
      <td>38.0</td>
      <td>facebook</td>
      <td>0</td>
      <td>en</td>
      <td>seo</td>
      <td>google</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Chrome</td>
      <td>NDF</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4ft3gnwmtx</td>
      <td>2010-09-28</td>
      <td>20090609231247</td>
      <td>2010-08-02</td>
      <td>FEMALE</td>
      <td>56.0</td>
      <td>basic</td>
      <td>3</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Windows Desktop</td>
      <td>IE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bjjt8pjhuk</td>
      <td>2011-12-05</td>
      <td>20091031060129</td>
      <td>2012-09-08</td>
      <td>FEMALE</td>
      <td>42.0</td>
      <td>facebook</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Firefox</td>
      <td>other</td>
    </tr>
    <tr>
      <th>4</th>
      <td>87mebub9p4</td>
      <td>2010-09-14</td>
      <td>20091208061105</td>
      <td>2010-02-18</td>
      <td>-unknown-</td>
      <td>41.0</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Chrome</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>




```python
''' 
    let's find out, If we have null values or not
    But, there is one more thing we need to keep in mind, from looking the training data analysis we observed
    that we have some '-unknown-' values in our Gender and first_browser feature which is clearly not a gender nor browser
    so we will be replacing this '-unknown-' value with NaN and later deal with it accordingly
'''
#Replacing "-unknown-" values with Null values
train_users['gender'].replace({'-unknown-':np.nan}, inplace = True)
train_users['first_browser'].replace({'-unknown-':np.nan}, inplace = True)

null_values = train_users.isnull().sum()

#Checking how many features having how much null values
print("**************** Null Values in Training Data **************** ")
for index in range(0, len(null_values)):
    if null_values[index] > 0:
        print('{:.2f} % ({} of {}) datapoints are NaN in "{}" feature'.format((null_values[index]/len(train_users))*100,
                                                             null_values[index], len(train_users), train_users.columns[index] )) 
```

    **************** Null Values in Training Data **************** 
    58.35 % (124543 of 213451) datapoints are NaN in "date_first_booking" feature
    44.83 % (95688 of 213451) datapoints are NaN in "gender" feature
    41.22 % (87990 of 213451) datapoints are NaN in "age" feature
    2.84 % (6065 of 213451) datapoints are NaN in "first_affiliate_tracked" feature
    12.77 % (27266 of 213451) datapoints are NaN in "first_browser" feature
    

1. **Here we can see that, we have major Missing values in our date_first_booking and age feature, we will be removing date_first_booking feature and for rest feature's missing values we will dealt accordingly**


```python
#Let's Check out if our data is balanced or not
print(train_users['country_destination'].value_counts())

print("\n{:.2f} % People decided to visit US or not to visit at all".format((train_users['country_destination'].value_counts()[['NDF','US']].sum()/len(train_users))*100))
#This Shows 87.56 Percent of the users either decided to Travel to US or decided not to Travel at all
```

    NDF      124543
    US        62376
    other     10094
    FR         5023
    IT         2835
    GB         2324
    ES         2249
    CA         1428
    DE         1061
    NL          762
    AU          539
    PT          217
    Name: country_destination, dtype: int64
    
    87.57 % People decided to visit US or not to visit at all
    

1. **Here we can notice that, dataset is highly imbalanced**
2. **Most of the user did not visit any destination and who decided to visit ,their destination was US**


```python
train_users.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>timestamp_first_active</th>
      <th>age</th>
      <th>signup_flow</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.134510e+05</td>
      <td>125461.000000</td>
      <td>213451.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.013085e+13</td>
      <td>49.668335</td>
      <td>3.267387</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.253717e+09</td>
      <td>155.666612</td>
      <td>7.637707</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.009032e+13</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.012123e+13</td>
      <td>28.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.013091e+13</td>
      <td>34.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.014031e+13</td>
      <td>43.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.014063e+13</td>
      <td>2014.000000</td>
      <td>25.000000</td>
    </tr>
  </tbody>
</table>
</div>



1. **Here we can observe , timestamp first active supposed to be in datetime format but it is not, and we have to convert it.**
2. **Second thing, we can observe, we have max age 2014 which is not possible and min age 1, which also not possbile, so we need to deal with them as well.**

####**date_account_created**


```python
#As this feature is releated to date object, first let's find out what exactly the type of this feature is in our dataset
#We are doing this because, if their type is not date or datetime, 
#we can convert them into date or datetime object and can use them much more effeciently
print('"date_account_created" type is   : ', type(train_users['date_account_created'][0]))

# Here we can see that feature is not having type as date or datetime, so we need to convert it into date feature 
# and we have already checked that this feature is not having any Null values so we are good to convert it
```

    "date_account_created" type is   :  <class 'str'>
    


```python
#Converting to Str to datetime object, and our date is in "YYYY-MM-DD" format
train_users['date_account_created'] = pd.to_datetime(train_users['date_account_created'])
```


```python
train_users_copy = train_users.sort_values(by='date_account_created')
train_users_copy['date_account_created_count'] = 1
train_users_copy['date_account_created_count'] = train_users_copy['date_account_created_count'].cumsum()
train_users_copy['year_account_created'] =  pd.DatetimeIndex(train_users_copy['date_account_created']).year

sns.set_style('whitegrid')
figure(figsize=(12,5))
sns.lineplot(data = train_users_copy, x = 'year_account_created', y = 'date_account_created_count')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff5128ed518>




![png](output_24_1.png)


1. **Here we're trying to find the trend of user creation, if user creation is increasing by year or not.**
2. **So, for that we fetched year column from our date_account_created column and also created a date_account_created_count column which consists comulative sum of accounts created to that particular date.**
3. **We can clearly see here number of accounts creation is almost exponentialy increasing every year.**


```python
#Lets plot the graph for country_destination, hued by Acounts_created_on_Weekdays and see if signup_method is affecting the choice of country
#Creating days Column from "timestamp_first_active" column

#Below one line of code is inspired by this https://stackoverflow.com/questions/29096381/num-day-to-name-day-with-pandas
train_users['Acounts_created_on_Weekdays'] = train_users['date_account_created'].apply(lambda day: dt.datetime.strftime(day, '%A'))
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue = 'Acounts_created_on_Weekdays',
              order = train_users['country_destination'].value_counts().index).set_yscale('log')
train_users.drop(['Acounts_created_on_Weekdays'], inplace = True, axis = 1)
```


![png](output_26_0.png)


1. **Here we can observe that Most of the accounts are created on weekdays and lesser accounts are created on weekends**
2. **Netherlend (NL), Portugul (PT), Spain (ES) and Italy (IT) has more number of accounts created on Monday.**
3. **We also can see that Portugul (PT), and Australia has less number of account created on Friday's.**
4. **We can say weekday's are helpful in predicting the country_destination.**


```python
#Lets plot the graph for country_destination, hued by Acounts_created_on_months and see if signup_method is affecting the choice of country
#Creating Year Column from "timestamp_first_active" column

#Below code is inspired by this https://stackoverflow.com/questions/29096381/num-day-to-name-day-with-pandas
train_users['Acounts_created_on_months'] = train_users['date_account_created'].apply(lambda month: dt.datetime.strftime(month, '%B'))
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue = 'Acounts_created_on_months', palette = "husl",
              order = train_users['country_destination'].value_counts().index).set_yscale('log')
train_users.drop(['Acounts_created_on_months'], inplace = True, axis = 1)
```


![png](output_28_0.png)


1. **We can see in May France and Australia has more bookings than other countries.**
2. **In December US and Australia has more bookings than other countries.**
3. **From Above Graph we can see that Month is helping in deciding the choice of country.**

####**gender**


```python
#Let's Plot the graph of Gender and their count, just to check if Gender is affecting the booking or not booking choice
sns.set_style('whitegrid')
figure(figsize = (12,5))
sns.countplot(data = train_users, x = 'gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe8715aa390>




![png](output_31_1.png)


1. **Here we can see that there is not much difference between Male and Female user count. but Female Users are slightly more in count**
2. **and we have major chunk of our Gender data as Null value, so we need to take care of this later.**
3. **Now let's check if Gender is affecting the choice of country or choice of doing booking or not.**


```python
#Let's plot the country_destination hued by gender, and check weather gender alone matters in predicting the country destination or not
sns.set_style('whitegrid')
figure(figsize = (12,5))
sns.countplot(data = train_users, x = 'country_destination', hue='gender').set_yscale('log')
```


![png](output_33_0.png)


1. **Here We can see that, it seems gender does not matter in choosing the country, Most of the people are likely to book for US or not to book at all.**
3. **So overall we can say that Gender is not much contributing to predict the country of choice of user.**


```python
#Checking if Gender and Age together are impacting the country choice
sns.set_style('whitegrid')
figure(figsize=(20, 8))
sns.boxplot(data = train_users, x = 'country_destination', y = 'age', hue= 'gender')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff502d73e48>




![png](output_35_1.png)


1. **Above Box Plot shows lot of information, we can se FR, CA, IT, DE and AU females are younger and males are older.**
2. **In Portugul and Australia Other category is almost nothing.**
3. **In United Kingdom (GB) Females are older and have more range.**

####**age**


```python
#As we have already seen above, that age column is having inconsistencies let's check them out
print("Minimum Age Given : ", train_users['age'].min()) 
print("Maximum Age Given : ", train_users['age'].max())
```

    Minimum Age Given :  1.0
    Maximum Age Given :  2014.0
    

1. **Here least age is 1 and max age is 2014 which is not possible, as minimum age requirement to book on airbnb is 18, check out the below link**
https://www.airbnb.co.in/help/article/2876/how-old-do-i-have-to-be-to-book-on-airbnb?locale=en&_set_bev_on_new_domain=1606819200_MDFhY2I5MzhlYjVm#:~:text=You%20must%20be%2018%20years,account%20to%20travel%20or%20host.
2. **Also the oldest person ever lived was of 122 Years, check out the below link**
https://www.guinnessworldrecords.com/news/2020/5/worlds-oldest-man-bob-weighton-dies-aged-112#:~:text=The%20oldest%20person%20ever%20to,days%20on%2012%20June%202013.



```python
#Now Let's checkout our age defaulters count
print("Users Count with Age > 122 : ", sum(train_users['age']>122))
print("Users Count with age < 18  : ", sum(train_users['age']<18))

'''
    One thing to notice as a common sense is, 122 or around, old person's chance to visit is very less 
    so we will take a number lesser than this (let's say 90) 
    and will replace all these values with Null for now, and later will use suitable technique to fill teses ages
'''

train_users['age'] = train_users['age'].apply(lambda age : np.nan if (age > 90 or age<18) else age)
```

    Users Count with Age > 122 :  781
    Users Count with age < 18  :  158
    


```python
#Let's checkout the distribution of the plot
sns.set_style("whitegrid")
sns.displot(train_users['age'].dropna(), kde = True, height = 5, aspect = 2.2, )
```




    <seaborn.axisgrid.FacetGrid at 0x7fe873e5d7b8>




![png](output_41_1.png)


1. **From the above plot we can observe that most of the users are from age 25-45 But this doesn't tells us much about weather this information is helpful or not.**
2. **Let's check out if age has some preference of selecting country or not**


```python
#Let's check if age is contributing to choice of country or not, that is our main motive
sns.set_style('whitegrid')
figure(figsize=(12, 6))
sns.boxplot(data = train_users, x = 'country_destination', y = 'age')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe8711419b0>




![png](output_43_1.png)


1. **From the above box plot we can see that, People who are older than 40 are more likely to travel France (FR), Canada (CA), United Kingdom (UK), Australia (AU) and Italy (IT), German (DE)**
2. **People who are lesser than 40 are more likely to go to Spain (ES), Netherlands (NL)**

####**signup_method, and signup_flow**


```python
#Lets plot the graph for country_destination, hued by signup_method and see if signup_method is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='signup_method', palette = "husl").set_yscale('log')
```


![png](output_46_0.png)


1. **Here we can see that United kingdom and Netherlend and Australia doesn't have signup_method as google.**

2. **So, This feature also be slightly helpful.**


```python
#Lets plot the graph for country_destination, hued by signup_flow and see if signup_flow is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='signup_flow', palette = "husl").set_yscale('log')
```


![png](output_48_0.png)


1. **Here we can see many noticable things, like if we see Netherlend we have 1 as signup flow but in Portugul we don't have signup flow, like this we have many examples.**

2. **So we can say that this is also helpfull in predicting the country.**


####**language**


```python
#Lets plot the graph for country_destination, hued by language and see if language is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'language', hue = 'country_destination').set_yscale('log')
```


![png](output_51_0.png)


1. **From the above graph we can see that, People who speaks Indonesian (id) and Croation (hr) made no bookings**

####**affiliate_channel, affiliate_provider, first_affiliate_tracked**


```python
#Lets plot the graph for country_destination, hued by affiliate_channel and see if affiliate_channel is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='affiliate_channel', palette = "husl").set_yscale('log')
```


![png](output_54_0.png)


1. **Here We can see that United Kingdom (GB) has content affiliate channel is least used and in almost all other countries, remarketing is least used.**
2. **In Canada (CA), we can see that SEO is less used than others, in all other countries case is reversed.**
3. **From the above graph we can conclude that affiliate channel is helping in predicting the country.**



```python
#Let's check weather affiliate channel and age together will help us to predicting the country_destination or not.
sns.set_style('whitegrid')
figure(figsize=(20, 8))
sns.boxplot(data = train_users, x = 'country_destination', y = 'age', hue = 'affiliate_channel')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff5008e3ef0>




![png](output_56_1.png)


1. **From above box plot we have so much information, like in GB poeple who used content as their affiliate_channel are seems to be older.**
2. **In Germany (DE), poeple with api as their affiliate_channel are seems to be younger.**
3. **People Portugul (PT) with sem-non-brand as their affiliate_channel seems to be older.**
4. **Overall we can say that, affiliate channel and age together is good predictor of country_destination.**


```python
#Lets plot the graph for country_destination, hued by affiliate_provider and see if affiliate_provider is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='affiliate_provider').set_yscale('log')
```


![png](output_58_0.png)


1. **Here also we can see many important things like, in US we have 'baidu', 'yandex', 'daum' provider used, but in other countreis like Canda, United Kingdom, Portugul etc we don't have these providers.**
2. **So overall this feature is also a helping hand in predicting the country_destination.** 


```python
#Lets plot the graph for country_destination, hued by first_affiliate_tracked and see if first_affiliate_tracked is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='first_affiliate_tracked').set_yscale('log')
```


![png](output_60_0.png)


1. **Here also we can observe many things, like in Spain(ES), Netherlends (NL), Germany (DE), Australia, we don't have marketing.**
2. **local ops, we have only in US and Australia.**
3. S**o overall we can say this affiliate_first_tracked is also a good predictor of country_destination.**

####**first_browser, first_device_type**


```python
#Lets plot the graph for country_destination, hued by first_browser and see if first_browser is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='first_browser', palette = "husl").set_yscale('log')
```


![png](output_63_0.png)


1. **Here we can see that, most of the country has used only limited number of browser, like if se Portugul, only chrome, firefox and IE are used as a first_browser.**
2. **We can se US has many first_browser categories unlike other countries.**
3. **This feature will suerly help to predict the country destination.**


```python
#Lets plot the graph for country_destination, hued by first_device_type and see if first_device_type is affecting the choice of country
sns.set_style('whitegrid')
figure(figsize = (20,8))
sns.countplot(data = train_users, x = 'country_destination', hue='first_device_type',).set_yscale('log')
```


![png](output_65_0.png)


1. **First Device type is also helpful in predicting the country_destination, because we can see many countries like United Kingdom, Spain, Portugul doesn't have Smart Phone as first_device_type.**
2. **We can in US Android Phone is more used than United Kingdom.**


```python
#Let' se if first_device_type and age together will help us in predicting the country_destination or not.
sns.set_style('whitegrid')
figure(figsize=(20, 8))
sns.boxplot(data = train_users, x = 'country_destination', y = 'age', hue = 'first_device_type')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff5014fd0f0>




![png](output_67_1.png)


1. **Here we can see that, in Australia people aged between 40 - 60 are mostly using Desktop (other) as their first_device_type, which is uncommon for other countries.**
2. **Also, In United Kingdom and Portugul iPad is mostly used by older people.**
3. **In Portugul Android Table is used by younger people and United Kingdom Android Phone are used by Younger People.**
4. **So we can say Age and First_device_type are good predictior of country_destination.**

####**country_destination**


```python
#Now Let's checkout the target_variable.
#First Let's checkout t

sns.set_style('whitegrid')
figure(figsize=(12, 5))
sns.countplot(data=train_users, x = 'country_destination')

#let's Print the Percentage of of count as well
#Below code is inspired by https://www.kaggle.com/krutarthhd/airbnb-eda-and-xgboost
for country_index in range(train_users['country_destination'].nunique()):
    plt.text(country_index,  train_users['country_destination'].value_counts()[country_index] + 1500, 
             str(round((train_users['country_destination'].value_counts()[country_index]/total_datapoints) * 100, 2)) + "%", ha = 'center')
```


![png](output_70_0.png)


1. **Above graph is supporting our perivous statement that, most of the users either decided to travel to US or decided not to Travell at all**

## Test, Sessions, Age_Gender, Countries datafiles

#### Test Data

**All the preprocessing steps would be same for testing data as Training data, also we have already done major EDA on Training data and as features are same so we won't be spending much time on EDA of Testing data.**


```python
test_users.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 62096 entries, 0 to 62095
    Data columns (total 15 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   id                       62096 non-null  object 
     1   date_account_created     62096 non-null  object 
     2   timestamp_first_active   62096 non-null  int64  
     3   date_first_booking       0 non-null      float64
     4   gender                   62096 non-null  object 
     5   age                      33220 non-null  float64
     6   signup_method            62096 non-null  object 
     7   signup_flow              62096 non-null  int64  
     8   language                 62096 non-null  object 
     9   affiliate_channel        62096 non-null  object 
     10  affiliate_provider       62096 non-null  object 
     11  first_affiliate_tracked  62076 non-null  object 
     12  signup_app               62096 non-null  object 
     13  first_device_type        62096 non-null  object 
     14  first_browser            62096 non-null  object 
    dtypes: float64(2), int64(2), object(11)
    memory usage: 7.1+ MB
    

1. **Here We can see that we have date_first_booking column has zero values, and this doesn't make sense to have booking date in test data**
2. **So, we will be dropping this, also we can see that we also have Null values in age and first_affiliate_tracked column**
3. **So we also need to use same techniques to fill these missing values and conversion of features as we will use in training data**


```python
#Let's Checkou the Head of the Testing data
test_users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date_account_created</th>
      <th>timestamp_first_active</th>
      <th>date_first_booking</th>
      <th>gender</th>
      <th>age</th>
      <th>signup_method</th>
      <th>signup_flow</th>
      <th>language</th>
      <th>affiliate_channel</th>
      <th>affiliate_provider</th>
      <th>first_affiliate_tracked</th>
      <th>signup_app</th>
      <th>first_device_type</th>
      <th>first_browser</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5uwns89zht</td>
      <td>2014-07-01</td>
      <td>20140701000006</td>
      <td>NaN</td>
      <td>FEMALE</td>
      <td>35.0</td>
      <td>facebook</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Moweb</td>
      <td>iPhone</td>
      <td>Mobile Safari</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jtl0dijy2j</td>
      <td>2014-07-01</td>
      <td>20140701000051</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Moweb</td>
      <td>iPhone</td>
      <td>Mobile Safari</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xx0ulgorjt</td>
      <td>2014-07-01</td>
      <td>20140701000148</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>linked</td>
      <td>Web</td>
      <td>Windows Desktop</td>
      <td>Chrome</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6c6puo6ix0</td>
      <td>2014-07-01</td>
      <td>20140701000215</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>linked</td>
      <td>Web</td>
      <td>Windows Desktop</td>
      <td>IE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>czqhjk3yfe</td>
      <td>2014-07-01</td>
      <td>20140701000305</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Safari</td>
    </tr>
  </tbody>
</table>
</div>




```python
''' 
    let's find out, If we have null values or not
    But, there is one more thing we need to keep in mind, from looking the Testing data analysis we observed
    that we have some '-unknown-' values in our Gender and first_browser feature which is clearly not a gender nor browser
    so we will be replacing this '-unknown-' value with NaN and later deal with it accordingly
'''
#Replacing "-unknown-" values with Null values
test_users['gender'].replace({'-unknown-':np.nan}, inplace = True)
test_users['first_browser'].replace({'-unknown-':np.nan}, inplace = True)

null_values = test_users.isnull().sum()

#Checking how many features having how much null values
print("**************** Null Values in Testing Data **************** ")
for index in range(0, len(null_values)):
    if null_values[index] > 0:
        print('{:.2f} % ({} of {}) datapoints are NaN in "{}" feature'.format((null_values[index]/len(test_users))*100,
                                                             null_values[index], len(test_users), test_users.columns[index] )) 
```

    **************** Null Values in Training Data **************** 
    100.00 % (62096 of 62096) datapoints are NaN in "date_first_booking" feature
    54.42 % (33792 of 62096) datapoints are NaN in "gender" feature
    46.50 % (28876 of 62096) datapoints are NaN in "age" feature
    0.03 % (20 of 62096) datapoints are NaN in "first_affiliate_tracked" feature
    27.58 % (17128 of 62096) datapoints are NaN in "first_browser" feature
    

1. **Here We can see date_first_booking has 100% Null values, which we have already discussed and will be removing this column from testing data.**
2. **Next things, we have missing values in gender, age, first_affiliate_tracked and first_browser.**
3. **We will be using same technique to fill these missing values which we will be using in Training data.**


```python
test_users.head()
#here also we can we have -unknown- value in gender and first_browser features.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date_account_created</th>
      <th>timestamp_first_active</th>
      <th>date_first_booking</th>
      <th>gender</th>
      <th>age</th>
      <th>signup_method</th>
      <th>signup_flow</th>
      <th>language</th>
      <th>affiliate_channel</th>
      <th>affiliate_provider</th>
      <th>first_affiliate_tracked</th>
      <th>signup_app</th>
      <th>first_device_type</th>
      <th>first_browser</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5uwns89zht</td>
      <td>2014-07-01</td>
      <td>20140701000006</td>
      <td>NaN</td>
      <td>FEMALE</td>
      <td>35.0</td>
      <td>facebook</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Moweb</td>
      <td>iPhone</td>
      <td>Mobile Safari</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jtl0dijy2j</td>
      <td>2014-07-01</td>
      <td>20140701000051</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Moweb</td>
      <td>iPhone</td>
      <td>Mobile Safari</td>
    </tr>
    <tr>
      <th>2</th>
      <td>xx0ulgorjt</td>
      <td>2014-07-01</td>
      <td>20140701000148</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>linked</td>
      <td>Web</td>
      <td>Windows Desktop</td>
      <td>Chrome</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6c6puo6ix0</td>
      <td>2014-07-01</td>
      <td>20140701000215</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>linked</td>
      <td>Web</td>
      <td>Windows Desktop</td>
      <td>IE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>czqhjk3yfe</td>
      <td>2014-07-01</td>
      <td>20140701000305</td>
      <td>NaN</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Safari</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
Now let's check one of the important thing, let's find out if we have categories of any features in testing data 
which we haven't seen in training data
'''

categorical_columns = ['gender','signup_method', 'first_browser',
                        'language', 'affiliate_channel', 'affiliate_provider',
                        'first_affiliate_tracked', 'signup_app', 'first_device_type',
                        ]

flag = True
for column in categorical_columns:
    if (train_users[column].nunique()  == test_users[column].nunique()):
        pass
    else:
        flag = False
        print('Categories are Not Same in {} in Training and Testing Data'.format(column))
if(flag):
    print("Categories in Testing and Training Data are same")

```

1. **Here We can see that, there is no category which presents in Testing data but not in Training data**

#### Sessions Data


```python
#Let's Checkout Sessions data
sessions.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10567737 entries, 0 to 10567736
    Data columns (total 6 columns):
     #   Column         Dtype  
    ---  ------         -----  
     0   user_id        object 
     1   action         object 
     2   action_type    object 
     3   action_detail  object 
     4   device_type    object 
     5   secs_elapsed   float64
    dtypes: float64(1), object(5)
    memory usage: 483.8+ MB
    


```python
sessions.head()
#First Let's find out the Null values in sessions data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>action</th>
      <th>action_type</th>
      <th>action_detail</th>
      <th>device_type</th>
      <th>secs_elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>319.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>d1mm9tcy42</td>
      <td>search_results</td>
      <td>click</td>
      <td>view_search_results</td>
      <td>Windows Desktop</td>
      <td>67753.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>301.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d1mm9tcy42</td>
      <td>search_results</td>
      <td>click</td>
      <td>view_search_results</td>
      <td>Windows Desktop</td>
      <td>22141.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>d1mm9tcy42</td>
      <td>lookup</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Windows Desktop</td>
      <td>435.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#let's find out, If we have null values or not
null_values = sessions.isnull().sum()

print("**************** Null Values in Sessions Data **************** ")
for index in range(0, len(null_values)):
    if null_values[index] > 0:
        print('{:.2f} % ({} of {}) datapoints are NaN in "{}" feature'.format((null_values[index]/len(sessions))*100,
                                                             null_values[index], len(sessions), sessions.columns[index] ))

```

    **************** Null Values in Sessions Data **************** 
    0.33 % (34496 of 10533241) datapoints are NaN in "user_id" feature
    0.76 % (79626 of 10533241) datapoints are NaN in "action" feature
    10.69 % (1126204 of 10533241) datapoints are NaN in "action_type" feature
    10.69 % (1126204 of 10533241) datapoints are NaN in "action_detail" feature
    1.29 % (136031 of 10533241) datapoints are NaN in "secs_elapsed" feature
    

1. **First observation is that we have 0.33% user ids are not given, so we have to drop all these values we don't have any other choice**
2. **But Also we can see that, we have one or more rows for one user id, so this is possible that all these missing user id doesn't make any difference Because data for that user may have already given.**
3. **Also, We don't have major missing values for other features in sessions data**


```python
#Let's Find out for how for how many training and testing users, session data is given
sessions_users_set = set(sessions['user_id'].dropna().unique())
train_users_set = set(train_users['id'].dropna().unique())
test_users_set  = set(test_users['id'].dropna().unique())

print("{:.2f}% of Train User's Sessions' Data is Available in Sessions File".format(len(sessions_users_set & train_users_set)/len(train_users)*100))
print("{:.2f}% of Test User's Sessions' Data is Available in Sessions File".format(len(sessions_users_set & test_users_set)/len(test_users)*100))


```

    34.58% of Train User's Sessions' Data is Available in Sessions File
    99.31% of Test User's Sessions' Data is Available in Sessions File
    

1. **Here we can clearly see that, we don't have sessions data for almost 65 % of training data points**
2. **So, here we hava to try to train our model with sessions data and without sessions data with pros and cons of each method**
3. **For Training data we can see, we have sessions data for more than 99 percent of data points.**

#### Age_Gender Data


```python
#Let's Checkout the age_gender_bkts file
age_gender.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 420 entries, 0 to 419
    Data columns (total 5 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   age_bucket               420 non-null    object 
     1   country_destination      420 non-null    object 
     2   gender                   420 non-null    object 
     3   population_in_thousands  420 non-null    float64
     4   year                     420 non-null    float64
    dtypes: float64(2), object(3)
    memory usage: 16.5+ KB
    


```python
age_gender.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_bucket</th>
      <th>country_destination</th>
      <th>gender</th>
      <th>population_in_thousands</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100+</td>
      <td>AU</td>
      <td>male</td>
      <td>1.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>95-99</td>
      <td>AU</td>
      <td>male</td>
      <td>9.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90-94</td>
      <td>AU</td>
      <td>male</td>
      <td>47.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85-89</td>
      <td>AU</td>
      <td>male</td>
      <td>118.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80-84</td>
      <td>AU</td>
      <td>male</td>
      <td>199.0</td>
      <td>2015.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
age_gender['year'].value_counts()
```




    2015.0    420
    Name: year, dtype: int64



1. **Here We can see that this data is given for year 2015, but we have data upto 2014 in train and test data files as we can in below cells**
2. **So, I am not sure how can i this file contribute in predicting the target variable, will explore more going ahead**


```python
print("Maximum date in Training data ", train_users['date_account_created'].max())
print("Maximum date in Testing data ", test_users['date_account_created'].max())
```

    Maximum date in Training data  2014-06-30
    Maximum date in Testing data  2014-09-30
    

#### Countries Data


```python
#Let's Checkout countries file
countries.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 7 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   country_destination            10 non-null     object 
     1   lat_destination                10 non-null     float64
     2   lng_destination                10 non-null     float64
     3   distance_km                    10 non-null     float64
     4   destination_km2                10 non-null     float64
     5   destination_language           10 non-null     object 
     6   language_levenshtein_distance  10 non-null     float64
    dtypes: float64(5), object(2)
    memory usage: 688.0+ bytes
    


```python
countries
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_destination</th>
      <th>lat_destination</th>
      <th>lng_destination</th>
      <th>distance_km</th>
      <th>destination_km2</th>
      <th>destination_language</th>
      <th>language_levenshtein_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AU</td>
      <td>-26.853388</td>
      <td>133.275160</td>
      <td>15297.7440</td>
      <td>7741220.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA</td>
      <td>62.393303</td>
      <td>-96.818146</td>
      <td>2828.1333</td>
      <td>9984670.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DE</td>
      <td>51.165707</td>
      <td>10.452764</td>
      <td>7879.5680</td>
      <td>357022.0</td>
      <td>deu</td>
      <td>72.61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ES</td>
      <td>39.896027</td>
      <td>-2.487694</td>
      <td>7730.7240</td>
      <td>505370.0</td>
      <td>spa</td>
      <td>92.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FR</td>
      <td>46.232193</td>
      <td>2.209667</td>
      <td>7682.9450</td>
      <td>643801.0</td>
      <td>fra</td>
      <td>92.06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GB</td>
      <td>54.633220</td>
      <td>-3.432277</td>
      <td>6883.6590</td>
      <td>243610.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>IT</td>
      <td>41.873990</td>
      <td>12.564167</td>
      <td>8636.6310</td>
      <td>301340.0</td>
      <td>ita</td>
      <td>89.40</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NL</td>
      <td>52.133057</td>
      <td>5.295250</td>
      <td>7524.3203</td>
      <td>41543.0</td>
      <td>nld</td>
      <td>63.22</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PT</td>
      <td>39.553444</td>
      <td>-7.839319</td>
      <td>7355.2534</td>
      <td>92090.0</td>
      <td>por</td>
      <td>95.45</td>
    </tr>
    <tr>
      <th>9</th>
      <td>US</td>
      <td>36.966427</td>
      <td>-95.844030</td>
      <td>0.0000</td>
      <td>9826675.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>



1. **Not Sure of this file, how can i use this but few ideas i can think of are as below**
    1. **I can add one column of distance in training data, because some people like to go farther and some likes to nearest destination**
    2. **Second I can add language_levenshtein_distance, because people likely to go where they can find people who speakes same language as they speak and if lavenshtein distance is small between person language and destination country language, so there would be more chance that person would like to go there**.

## EDA Conclusion

1. **Data is Highly Imbalanced**
2.**We Have Null Values in almost 4-5 Features and Need to clean few features like "first_browser" and "Gender" has "-unknown-" as values and have unexpected values in "age" feature.**
3. **Most of the users Either Decided to Travel to US or decided not to travel at all.**
4. **Most of the accounts are created in June and on Monday's, Tueday's and Wednesday's**
5. **Most of the Users are age between 20-45 and Female users are slightly more than Male Users**
6. **Data for Users are given between year 2009 and 2014 and each User accounts are increasing with good amount of growth**
7. **There is no column or category which is in Training data but not in Testing data, so no worries to deal with surprised cateogries.**
8. **Most of the users used their signup method as basic**
9. **English Language is the most Popular and used almost by more than 96% users**
10. **Chrome Safari and Firefox are mostly Used browsers**
11. **"date_first_booking" Column is given in Testing data with 0 values, which is of no use and will be dropping this column**
12. **We have 0.33% users IDs are Null in Sessions file and almost 65% of users of Training data are missing in Sessions file**
13. **More than 99% testing data is having Sessions data**


```python

```

## Modeling


```python
'''First Let's combine both the dataset, train and test and then perform all data preproccesing steps and Encodings.
   I beleive this would cause data leakage Problem, but as we are solving kaggle compettion, I need to focus more 
   on getting highest score, Please suggest if i should not do such thing ? '''

train_test = pd.concat(((train_users.drop(['id', 'country_destination', 'date_first_booking'], axis= 1)), 
                             (test_users.drop(['id', 'date_first_booking'], axis= 1))), axis = 0)
```


```python
'''For the First Try, I have taken everthing simple, no complex impuations, all just straight forward.
   Going ahead we'll try more advance approached for imputations.
   Here I am just, Imputing Null values and dealing with some unwanted values in columns'''

#creating object of SimpleImputer
imputer_cat = SimpleImputer(strategy='most_frequent')
imputer_num = SimpleImputer()

#First doing some data cleaning
train_test['gender'].replace({'-unknown-':np.nan}, inplace = True)
train_test['first_browser'].replace({'-unknown-':np.nan}, inplace = True)
train_test['age'] = train_test['age'].apply(lambda age : np.nan if (age > 90 or age<18) else age)

#Doing Imputation of gender, first_browser, first_affiliate_tracked, age
train_test['gender'] = imputer_cat.fit_transform(train_test['gender'].values.reshape(-1, 1))
train_test['first_browser'] = imputer_cat.fit_transform(train_test['first_browser'].values.reshape(-1, 1))
train_test['first_affiliate_tracked'] = imputer_cat.fit_transform(train_test['first_affiliate_tracked'].values.reshape(-1, 1))
train_test['age'] = imputer_num.fit_transform(train_test['age'].values.reshape(-1, 1))
```


```python
'''First we will be using date_account_created feature only, for that also we will create
   3 new features dac_day, dac_month, dac_year.'''

#First Converting feature into datetime object and then creating other features
train_test['date_account_created'] = pd.to_datetime(train_test['date_account_created'])
train_test['dac_day'] =  train_test['date_account_created'].apply(lambda date : date.day)
train_test['dac_month'] =  train_test['date_account_created'].apply(lambda date : date.month)
train_test['dac_year'] =  train_test['date_account_created'].apply(lambda date : date.year)
```


```python
''' For now i will work only with these features just for simplicity, and later will increase complexity gradually
        'dac_day', 'dac_month', 'dac_year', 'signup_flow', 'age', 
        'signup_method', 'gender',
        'language', 'affiliate_channel', 'affiliate_provider',
        'first_affiliate_tracked', 'signup_app', 'first_device_type',
        'first_browser' '''

#dealing with categorical_variables
ohe = OneHotEncoder()

signup_method_ohe = ohe.fit_transform(train_test['signup_method'].values.reshape(-1,1)).toarray()
gender_ohe = ohe.fit_transform(train_test['gender'].values.reshape(-1,1)).toarray()
language_ohe = ohe.fit_transform(train_test['language'].values.reshape(-1,1)).toarray()
affiliate_channel_ohe = ohe.fit_transform(train_test['affiliate_channel'].values.reshape(-1,1)).toarray()
affiliate_provider_ohe = ohe.fit_transform(train_test['affiliate_provider'].values.reshape(-1,1)).toarray()
first_affiliate_tracked_ohe = ohe.fit_transform(train_test['first_affiliate_tracked'].values.reshape(-1,1)).toarray()
signup_app_ohe = ohe.fit_transform(train_test['signup_app'].values.reshape(-1,1)).toarray()
first_device_type_ohe = ohe.fit_transform(train_test['first_device_type'].values.reshape(-1,1)).toarray()
first_browser_ohe = ohe.fit_transform(train_test['first_browser'].values.reshape(-1,1)).toarray()

#Getting teh labels for Target Classs
le = LabelEncoder()
y_train_le = le.fit_transform(train_users['country_destination'])
```


```python
#Now Just Combining All the Independent Features and for modeling
train_test_values = np.concatenate((signup_method_ohe, gender_ohe, language_ohe, affiliate_channel_ohe,
                     affiliate_provider_ohe, first_affiliate_tracked_ohe, signup_app_ohe,
                    first_device_type_ohe, first_browser_ohe, train_test['dac_day'].values.reshape(-1, 1),
                    train_test['dac_month'].values.reshape(-1, 1), train_test['dac_year'].values.reshape(-1, 1),
                    train_test['signup_flow'].values.reshape(-1, 1), train_test['age'].values.reshape(-1, 1)),
                    axis = 1)
```


```python
#Her we're just splitting our training and test datapoints
X = train_test_values[:train_users.shape[0]]
X_test_final = train_test_values[train_users.shape[0]:]
X.shape, y_train_le.shape, X_test_final.shape
```




    ((213451, 138), (213451,), (62096, 138))



### NDCG Score Calculation : 
I have taken below function [NDCG Scorer](https://www.kaggle.com/davidgasquez/ndcg-scorer) Kaggle Kernel, **I am not sure if i can use this function in my Notebook, for now i just used it, please guide me if i need to write such Function myself?**


```python
"""Metrics to compute the model performance."""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)
```

### Stacking Model


```python
# First splitting our training data into 80-20 train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y_train_le , test_size = 0.2, random_state = 10, stratify = y_train_le)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#Now let's divide our dataset into 2 equal parts 50 - 50
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X_train, y_train , test_size = 0.5, random_state = 10, stratify = y_train)
print(X_train_50.shape, y_train_50.shape, X_test_50.shape, y_test_50.shape)
```

    (170760, 138) (170760,) (42691, 138) (42691,)
    (85380, 138) (85380,) (85380, 138) (85380,)
    


```python
''' Approach : I will work on X_train_50 and y_train_50, 
    from this dataset i will be creating 10 datasets with sampling with replacement.
    Now i will train 10 models on each of these datasets, and will predict on X_test_50
    will all these 10 model, now i will be having 10 columns of predictions of each Model and i will make the dataset of 
    these predictions and y_test_50 as target variable. and this model will be my meta classifier or final model. '''

#"random_samples_generator" this function basically generates the indexes of raondom samples with replacement
def random_samples_generator():
    """ 
    Generating 60 % Unique Indexes of our total data, 
    and in next step we will generate 40% of Indexes of our total data, from this 60% indexes
    with replacement.
    "Its Your Choice, weather you wanna take first 60% Unique data points or not", 
    you can take all data with duplicate data points if you want.    
    
    """   
    #Below two lines of code performs row sampling
    X_sample_indexes = np.random.choice(np.arange(len(X_train_50)), 
                                        size = int(len(X_train_50)/100 * 60), replace = False)

    #Generating 40% Indexes from above 60% of Indexes with duplicate indexes
    X_sample_indexes = np.append(X_sample_indexes, np.random.choice(X_sample_indexes, 
                                size = int(len(X_train_50)/100 * 60)))
    
    #Below lines of code is used for column sampling
    #Now Generating a Random Variable between 80(included) and 139(excluded)
    #Which is basically Number Columns We are gonna take for current Sample
    random_columns = np.random.randint(80, 139)

    #Now Column Sampling is being done
    sample_columns = np.random.choice(np.arange(138), size = random_columns, replace = False)

    return X_sample_indexes, sample_columns
```


```python
#Now We will be loading the saved base models and meta models and will be predicting on the the test data
#All the base models and meta model are trained in other note_book named as "base_models.ipynb"

!cp /content/drive/MyDrive/Study/"Case Study 1"/base_models -r /content
!cp /content/drive/MyDrive/Study/"Case Study 1"/base_model_cols.csv /content/
```


```python
#We will be using this file to fetch the column on which our base model was trained and will be 
#using same set of column in our testing set
base_model_cols = pd.read_csv('base_model_cols.csv')

#This file will contain the predictions of base models and will be used by metamodel for final predictions
base_model_test_preds = pd.DataFrame()
```


```python
#Loading the base models in doing predictions and saving in base_model_test_preds for metaclassifier to predict
path = '/content/base_models/'
base_models = os.listdir(path)

#Here We are just simply loading base model one by one and then loading columns on which the base model trained earlier
#then simply predicting on test data with same columns
for model_name in base_models:
    base_model = pickle.load(open(path+model_name, 'rb'))
    columns = [int(x) for x in base_model_cols[model_name.split('.')[0]][0].split(',')]
    base_model_test_preds[model_name] = base_model.predict(X_test_final[:, columns])
```


```python
#Now loading the meta model and predicting the probabilites for final predictions
# !cp /content/drive/MyDrive/Study/"Case Study 1"/meta_xgb.sav /content
meta_xgb = pickle.load(open('/content/meta_xgb.sav', 'rb'))
y_preds = meta_xgb.predict_proba(np.array(base_model_test_preds))
```


```python
y_preds = xgb_clf.predict_proba(X_test_final)
```


```python
'''This Code is basically used get top 5 predictions for the submission file
   Here we're just zipping predictions and classes together and sorting with predictions,
   and then taking top5 countries, that's it.'''

prediction_classes = le.classes_
user_list = []
predictions_list = []
for user_index in range(len(test_users)):
    user_list.extend([test_users['id'][user_index]] * 5)
    sorted_values = sorted(zip(y_preds[user_index], prediction_classes), reverse = True)[:5]
    predictions_list.extend([country[1] for country in sorted_values])
    
submission_file = pd.DataFrame({'id':user_list, 'country':predictions_list})
submission_file.to_csv('submission_stacking_4.csv', index = False)
```

We got 85.614 Accuracy with this method

![image.png](data:image/png;base64,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)


```python
import xgboost as xgb
```


```python

```


```python

```
