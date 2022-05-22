# import libraries
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Q1. Downloading the data
# We'll use the same NYC taxi dataset, but instead of "Green Taxi Trip Records", we'll use "For-Hire Vehicle Trip Records".
# Download the data for January and February 2021.
# Note that you need "For-Hire Vehicle Trip Records", not "High Volume For-Hire Vehicle Trip Records".
# Read the data for January. How many records are there?
df_jan = pd.read_parquet('datasets/fhv_tripdata_2021-01.parquet')
#df_jan.shape[0]
# A1. 1154112


# Q2.Computing duration
# Now let's compute the duration variable. It should contain the duration of a ride in minutes.
# What's the average trip duration in January?
df_jan['duration'] = df_jan['dropOff_datetime'] - df_jan['pickup_datetime']
df_jan['duration'] = df_jan['duration'].apply(lambda td: td.total_seconds() / 60)
#df_jan['duration'].mean()
# A2. 19.16


# Data preparation
# Check the distribution of the duration variable. There are some outliers.
# Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).
# How many records did you drop?
df_jan = df_jan[(df_jan.duration >= 1) & (df_jan.duration <= 60)]
#df_jan.shape[0]
# 1154112 - 1109826 = 44286


# Q3. Missing values
# The features we'll use for our model are the pickup and dropoff location IDs.
# But they have a lot of missing values there. Let's replace them with "-1".
# What's the fractions of missing values for the pickup location ID? I.e. fraction of "-1"s after you filled the NAs.
df_jan.PUlocationID.fillna(-1, inplace=True)
df_jan.DOlocationID.fillna(-1, inplace=True)

df_jan.PUlocationID.value_counts(normalize=True).apply('{0:.3f}'.format)
# A.3 %83.5


# Q4. One-hot encoding
# Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model.
#     Turn the dataframe into a list of dictionaries
#     Fit a dictionary vectorizer
#     Get a feature matrix from it
# What's the dimensionality of this matrix? (The number of columns).
categorical = ['PUlocationID', 'DOlocationID']
df_jan[categorical] = df_jan[categorical].astype(str)
train_dicts = df_jan[categorical].to_dict(orient='records')

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
#X_train.shape[1]  # 525
# A.4 525



# Q5. Training a model
# Now let's use the feature matrix from the previous step to train a model.
#     Train a plain linear regression model with default parameters
#     Calculate the RMSE of the model on the training data
# What's the RMSE on train?

y_train = df_jan['duration'].values
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_train)
mean_squared_error(y_train, y_pred, squared=False)
# A4. 10.52


# Q6. Evaluating the model
# Now let's apply this model to the validation dataset (Feb 2021).
# What's the RMSE on validation?

df_feb = pd.read_parquet('datasets/fhv_tripdata_2021-01.parquet')
df_feb['duration'] = df_feb['dropOff_datetime'] - df_feb['pickup_datetime']
df_feb['duration'] = df_feb['duration'].apply(lambda td: td.total_seconds() / 60)
df_feb = df_feb[(df_feb.duration >= 1) & (df_feb.duration <= 60)]
df_feb.PUlocationID.fillna(-1, inplace=True)
df_feb.DOlocationID.fillna(-1, inplace=True)
df_feb[categorical] = df_feb[categorical].astype(str)
val_dicts = df_feb[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_val = dv.fit_transform(val_dicts)
y_val = df_feb['duration'].values
y_pred = lr.predict(X_val)
mean_squared_error(y_val, y_pred, squared=False)
# A6. 11.01



