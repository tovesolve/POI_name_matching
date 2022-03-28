# dataset source: https://catalog.data.gov/dataset/community-points-of-interest

import datetime
import json
import math
import pandas as pd

data_file = 'points-of-interest.csv'
df = pd.read_csv(data_file, sep=";")

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

drop_attributes = ['owner', 'featurecode', 'owntype', 'fulladdr', 'descript', 'geo_shape']
df.drop(drop_attributes, axis=1, inplace=True)


# Split attribute geo_point_2d to two new attributes, longitude and latitude
splitter = lambda x: pd.Series([i for i in (str(x).split(', '))])
coordinates = df['geo_point_2d'].apply(splitter)
coordinates.rename(columns={0:'latitude',1:'longitude'},inplace=True)
coordinates = coordinates[['latitude', 'longitude']]
df['latitude'] = coordinates['latitude']      # add column longitude to the df
df['longitude'] = coordinates['longitude']      # add column latitude to the df
df.drop(['geo_point_2d'], axis=1, inplace=True)

print(df)
print(df.shape[0])
print(df.shape[1])
print(df.columns)
print(df.info())

df.to_pickle('./df_gov_nc' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl')



max = -math.inf
min = math.inf
for row in df['latitude']:
    if float(row) > max:
        max = float(row)
    if float(row) < min:
        min = float(row)

print("max latitude: ", max)
print("Min latitude: ", min)

max = -math.inf
min = math.inf
for row in df['longitude']:
    if float(row) > max:
        max = float(row)
    if float(row) < min:
        min = float(row)

print("max longitude: ", max)
print("Min longitude: ", min)