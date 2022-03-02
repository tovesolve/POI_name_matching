"""
Script for labeling data.

Script allows user to manually label Points of Interest(POIs) from two different datasets. The script show the user two closely located POIs and the user can with an input value
decide if the two POIs are a match or not.

Functions
---------
    * close_pairs - Give user closely located POIs to label and saves labelled data to a DataFrame
    * distance_meters - Calculate distance between two POIs
    * restrict_dataset - Filter DataFrame according to coordinates
    * main_florida - load POI datasets and start labeling
    * main_boston - load POI datasets and start labeling

Run script:                     python3 label_data.py
Run script without warnings:    python3 -W ignore label_data.py

"""

import datetime
import pandas as pd
from sqlite3 import Timestamp
from math import radians, sin, cos, sqrt, atan2
import mpu

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe


def close_pairs(df_osm, df_yelp, distance):
    ''' Label POIs.
    Selects closely located pairs from two datasets, label them: 0-no match, 1-match, 2-unsure, 3-not POIs.
    Saves pairs to dataframe.

    Parameters
    ----------
    df_osm : pandas DataFrame
        Dataset from OSM data.
    df_yelp : pandas DataFrame
        Dataset from Yelp data.
    distance : float
        Defines how far away a POI can be from another in any of the four directions (north, south, west, east) to be considered as a possible pair.

    Returns
    -------
    df_pairs
        a DataFrame of the labelled pairs
    '''

    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match'] #columns for the labelled dataset
    df_pairs = pd.DataFrame(columns=data_colnames)

    for index, poi_osm in df_osm.iterrows(): #iterate through POIs in OSM dataset
        lat_osm = float(poi_osm['latitude'])
        lon_osm = float(poi_osm['longitude'])

        for i, poi_yelp in df_yelp.iterrows(): #iterate through POIs in yelp dataset
            lat_yelp = float(poi_yelp['latitude'])
            lon_yelp = float(poi_yelp['longitude'])

            #restrict only compare POIs if they are closely located
            if (lat_yelp < lat_osm+distance and lat_yelp > lat_osm-distance) and (lon_yelp < lon_osm+distance and lon_yelp > lon_osm-distance):     
                alreadyLabelled = False
                #check if pair already exist in labelled data:
                for i, labelled_poi in df_pairs.iterrows():
                    if labelled_poi['osm_name'] == poi_osm['name'] and labelled_poi['yelp_name'] == poi_yelp['name']:
                        alreadyLabelled = True
                        print('already labelled: ', labelled_poi['osm_name'], poi_osm['name'], labelled_poi['yelp_name'], poi_yelp['name'])
                        break
                        
                if not alreadyLabelled:        
                    poi_dist = distance_meters(lat_osm, lon_osm, lat_yelp, lon_yelp) #calculate distance between POIs
                    
                    print(poi_osm['name'], " , ", poi_yelp['name'], ' distance between: ', poi_dist) #Print pair for user to consider
                    if poi_osm['name'].lower() == poi_yelp['name'].lower(): #if names are an exact match
                        df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 1}, ignore_index=True)
                    else:
                        while True:
                            str_num = input() #Take input from user
                            try:
                                num = int(str_num)
                                if num not in range(0,4): #restrict input to 0/1
                                    print("Only numbers 0, 1, 2 and 3 valid")
                                else:
                                    break
                            except ValueError:
                                print("not int, try again")

                        #Add pair to DataFrame:
                        if num == 1:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 1}, ignore_index=True)
                        elif num == 0:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 0}, ignore_index=True)
                        elif num == 2:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 2}, ignore_index=True)
                        elif num == 3:
                            df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 3}, ignore_index=True)

    df_pairs.to_pickle('./df_pairs_boston' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    return df_pairs


def distance_meters(lat1, lon1, lat2, lon2):
    '''
    Calculate distance between two places (POI1, POI2) on earth.

    Parameters
    ----------
    lat1 : float
        Latitude of POI1
    lon1 : float
        Longitude of POI1
    lat2 : float
        Latitude of POI2
    lon2 : float
        Longitude of POI2
    
    Returns
    -------
    dist
        the distance between the points in meters
    '''
    dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
    dist = dist * 1000 # convert distance to meters
    #print(dist)
    return dist

def restrict_dataset(df, top, bottom, left, right):
    '''
    Filter the DataFrame of POIs according to coordinates.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to filter
    top : float
        Latitude coordinate limit area from north.
    bottom : float
        Latitude coordinate limit area from south.
    left : float
        Latitude coordinate limit area from west.
    right : float
        Latitude coordinate limit area from right.
    
    Returns
    -------
    df
        The filtered DataFrame
    '''
    df = df[pd.to_numeric(df['latitude']) < top]
    df = df[pd.to_numeric(df['latitude']) > bottom]
    df = df[pd.to_numeric(df['longitude']) > left]
    df = df[pd.to_numeric(df['longitude']) < right]
    return df
    

def main_florida():
    '''
    Main method for Florida area
    '''
    df_osm = pd.read_pickle('df_osm_florida2022-02-25.094449.pkl')   #read dataframe osm data
    df_yelp = pd.read_pickle('df_yelp_florida2022-02-25.093315.pkl') #read dataframe yelp data

    df_osm = restrict_dataset(df_osm, 28.616707, 28.567949, -81.405150, -81.315333) 
    df_yelp = restrict_dataset(df_yelp, 28.616707, 28.567949, -81.405150, -81.315333)
    print('Number of rows in labelled df_osm: ', df_osm.shape[0])
    print('Number of rows in labelled df_yelp: ', df_yelp.shape[0])

    df = close_pairs(df_osm, df_yelp, distance=0.0002)  #0.001=111m
    print("Dataframe pairs:")
    print(df)
    print('Number of rows in labelled df: ', df.shape[0])


def main_boston():
    '''
    Main method for Boston area
    '''
    df_osm = pd.read_pickle('df_osm_ma2022-02-25.104802.pkl')   #read dataframe osm data
    df_yelp = pd.read_pickle('df_yelp_ma2022-02-25.103519.pkl') #read dataframe yelp data

    df_osm = restrict_dataset(df_osm, 42.366625, 42.349203, -71.068092, -71.055022) 
    df_yelp = restrict_dataset(df_yelp, 42.366625, 42.349203, -71.068092, -71.055022)
    print('Number of rows in osm df: ', df_osm.shape[0])
    print('Number of rows in yelp df: ', df_yelp.shape[0])

    df = close_pairs(df_osm, df_yelp, distance=0.0002)  #0.001=111m
    print("Dataframe pairs:")
    print(df)
    print('Number of rows in labelled df: ', df.shape[0])


def main():
    main_florida()
    #main_boston()

if __name__ == "__main__":
    main()