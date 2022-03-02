import datetime
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
df = df[df['state'] == 'MA'] #change to area in yelp data set that we want to filter on.

df.to_pickle('./df_yelp_florida' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl')
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df) #.loc[df['city'] == 'Altamonte Springs']
print('Number of rows in yelp: ', df.shape[0])