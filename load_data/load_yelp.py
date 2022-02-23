import json
import pandas as pd
data_file = open("yelp_academic_dataset_business.json")
data = []
for line in data_file:
    data.append(json.loads(line))
data_file.close()

df = pd.DataFrame(data)
drop_attributes = ['address', 'postal_code', 'stars', 'review_count', 'is_open', 'attributes', 'categories', 'hours']
df.drop(drop_attributes, axis=1, inplace=True)
df = df[df['state'] == 'FL']


df = df[pd.to_numeric(df['latitude']) < 28.591826]
df = df[pd.to_numeric(df['latitude']) > 28.547067]
df = df[pd.to_numeric(df['longitude']) > -81.372633]
df = df[pd.to_numeric(df['longitude']) < -81.303579]

df.to_pickle('./df_yelp_florida.pkl')
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df) #.loc[df['city'] == 'Altamonte Springs']