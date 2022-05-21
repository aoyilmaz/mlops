# import libraries
import pandas as pd

# Q1. Downloading the data
# We'll use the same NYC taxi dataset, but instead of "Green Taxi Trip Records", we'll use "For-Hire Vehicle Trip Records".
# Download the data for January and February 2021.
# Note that you need "For-Hire Vehicle Trip Records", not "High Volume For-Hire Vehicle Trip Records".
# Read the data for January. How many records are there?
df = pd.read_parquet('datasets/fhv_tripdata_2021-01.parquet')
df.shape[0]
# A1. 1154112


# Q2.Computing duration
# Now let's compute the duration variable. It should contain the duration of a ride in minutes.
# What's the average trip duration in January?









