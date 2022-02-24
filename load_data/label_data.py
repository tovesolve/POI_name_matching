import datetime
import pandas as pd
from sqlite3 import Timestamp
from math import radians, sin, cos, sqrt, atan2
import mpu

# Run script without warnings: python3 -W ignore test_compare.py

df_osm = pd.read_pickle('df_osm_florida.pkl')   #read dataframe osm data
df_yelp = pd.read_pickle('df_yelp_florida.pkl') #read dataframe yelp data

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe


'''
Label closely located pairs and save to dataframe.
'''
def close_pairs(df_osm, df_yelp, distance):
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match']
    df_pairs = pd.DataFrame(columns=data_colnames)
    for index, poi_osm in df_osm.iterrows():
        lat_osm = float(poi_osm['latitude'])
        lon_osm = float(poi_osm['longitude'])

        for i, poi_yelp in df_yelp.iterrows():
            lat_yelp = float(poi_yelp['latitude'])
            lon_yelp = float(poi_yelp['longitude'])

            #restrict comparisons
            if (lat_yelp < lat_osm+distance and lat_yelp > lat_osm-distance) and (lon_yelp < lon_osm+distance and lon_yelp > lon_osm-distance):
                poi_dist = distance_meters(lat_osm, lon_osm, lat_yelp, lon_yelp) #calculate distance betwenn POIs
                print(poi_osm['name'], " , ", poi_yelp['name'], ' distance between: ', poi_dist)

                if poi_osm['name'].lower() == poi_yelp['name'].lower():
                    df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': True}, ignore_index=True)
                else:
                    while True:
                        str_num = input()
                        num = int(str_num)
                        if num not in range(0,2): #restrict input to 0/1
                            print("Only numbers 0 and 1 valid")
                        else:
                            break

                    if num == 1:
                        df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': True}, ignore_index=True)
                        print('pair added')
                    elif num == 0:
                        df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': False}, ignore_index=True)
                        print('pair added')

    df_pairs.to_pickle('./df_pairs_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe
    return df_pairs

'''
Calculate distance between two places on earth.
'''
def distance_meters(lat1, lon1, lat2, lon2):
    dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    dist = dist * 1000 # convert to meters
    #print(dist)
    return dist



def main():
    df = close_pairs(df_osm, df_yelp, distance=0.002)  #0.001=111m
    print("Dataframe pairs:")
    print(df)
    print('Number of rows in labeled df: ', df.shape[0])

if __name__ == "__main__":
    main()