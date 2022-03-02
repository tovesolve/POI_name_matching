import datetime
import pandas as pd
from sqlite3 import Timestamp
from math import radians, sin, cos, sqrt, atan2
import mpu

# Run script without warnings: python3 -W ignore label_data.py

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe


'''
Label closely located pairs and save to dataframe.
'''
def close_pairs(df_osm, df_yelp, distance):
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match']
    df_pairs = pd.DataFrame(columns=data_colnames)
    checked_names = []
    for index, poi_osm in df_osm.iterrows():
        lat_osm = float(poi_osm['latitude'])
        lon_osm = float(poi_osm['longitude'])

        for i, poi_yelp in df_yelp.iterrows():
            lat_yelp = float(poi_yelp['latitude'])
            lon_yelp = float(poi_yelp['longitude'])

            #restrict comparisons
            if (lat_yelp < lat_osm+distance and lat_yelp > lat_osm-distance) and (lon_yelp < lon_osm+distance and lon_yelp > lon_osm-distance):     
                alreadyLabeled = False
                #om det finns en rad i df_pairs som redan innnehåller exakta paret.
                for i, labeled_poi in df_pairs.iterrows():
                    #print('checking:', labeled_poi['osm_name'], poi_osm['name'], labeled_poi['yelp_name'], poi_yelp['name'])
                    if labeled_poi['osm_name'] == poi_osm['name'] and labeled_poi['yelp_name'] == poi_yelp['name']:
                        alreadyLabeled = True
                        print('already labeled', labeled_poi['osm_name'], poi_osm['name'], labeled_poi['yelp_name'], poi_yelp['name'])
                        break
                        
                if not alreadyLabeled:        
                    poi_dist = distance_meters(lat_osm, lon_osm, lat_yelp, lon_yelp) #calculate distance betwenn POIs
                    print(poi_osm['name'], " , ", poi_yelp['name'], ' distance between: ', poi_dist)

                    if poi_osm['name'].lower() == poi_yelp['name'].lower():
                        df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 1}, ignore_index=True)
                    else:
                        while True:
                            str_num = input()
                            try:
                                num = int(str_num)
                                if num not in range(0,4): #restrict input to 0/1
                                    print("Only numbers 0, 1, 2 and 3 valid")
                                else:
                                    break
                            except ValueError:
                                print("not int, try again")

                        if num == 1:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 1}, ignore_index=True)
                            #print('pair added')
                        elif num == 0:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 0}, ignore_index=True)
                            #print('pair added')
                        elif num == 2:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 2}, ignore_index=True)
                            #print('pair added')
                        elif num == 3:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 3}, ignore_index=True)
                            #print('pair added')

    df_pairs.to_pickle('./df_pairs_boston' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe
    return df_pairs

'''
Calculate distance between two places on earth.
'''
def distance_meters(lat1, lon1, lat2, lon2):
    dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    dist = dist * 1000 # convert to meters
    #print(dist)
    return dist

def restrict_dataset(df, top, bottom, left, right):
    df = df[pd.to_numeric(df['latitude']) < top]
    df = df[pd.to_numeric(df['latitude']) > bottom]
    df = df[pd.to_numeric(df['longitude']) > left]
    df = df[pd.to_numeric(df['longitude']) < right]
    return df
    

def main_florida():
    df_osm = pd.read_pickle('df_osm_florida2022-02-25.094449.pkl')   #read dataframe osm data
    df_yelp = pd.read_pickle('df_yelp_florida2022-02-25.093315.pkl') #read dataframe yelp data

    df_osm = restrict_dataset(df_osm, 28.616707, 28.567949, -81.405150, -81.315333) # 28.591826, 28.567067, -81.372633, -81.303579
    df_yelp = restrict_dataset(df_yelp, 28.616707, 28.567949, -81.405150, -81.315333) # 28.591826, 28.567067, -81.372633, -81.303579
    print('Number of rows in labeled df_osm: ', df_osm.shape[0])
    print('Number of rows in labeled df_yelp: ', df_yelp.shape[0])

    df = close_pairs(df_osm, df_yelp, distance=0.0002)  #0.001=111m
    print("Dataframe pairs:")
    print(df)
    print('Number of rows in labeled df: ', df.shape[0])


def main_boston():
    df_osm = pd.read_pickle('df_osm_ma2022-02-25.104802.pkl')   #read dataframe osm data
    df_yelp = pd.read_pickle('df_yelp_ma2022-02-25.103519.pkl') #read dataframe yelp data

    #boston coordinates: (42.366625, 42.349203, -71.068092, -71.055022)

    df_osm = restrict_dataset(df_osm, 42.366625, 42.349203, -71.068092, -71.055022) # 28.591826, 28.567067, -81.372633, -81.303579
    df_yelp = restrict_dataset(df_yelp, 42.366625, 42.349203, -71.068092, -71.055022) # 28.591826, 28.567067, -81.372633, -81.303579
    print('Number of rows in osm df: ', df_osm.shape[0])
    print('Number of rows in yelp df: ', df_yelp.shape[0])

    df = close_pairs(df_osm, df_yelp, distance=0.0002)  #0.001=111m
    print("Dataframe pairs:")
    print(df)
    print('Number of rows in labeled df: ', df.shape[0])


def main():
    main_florida()
    #main_boston()

if __name__ == "__main__":
    main()