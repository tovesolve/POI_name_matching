import argparse
import pandas as pd
from sqlite3 import Timestamp
import time
import datetime

from drop_label import drop_rows_with_label

"""
A script used to a edit current label on an already labelled pair of POIs in a labelled dataframe from a pkl-file .
The script reads a current dataframe, edits the chosen label and overwrites the old dataframe and saves it as a pickled file with the same file name.

Run script using: 'python3 edit_label.py --df {pickled dataframe file} --osm_name "{osm_name}" --yelp_name "{yelp_name}" --new_value {int_value}'
Example: 'python3 edit_label.py --df df_pairs_boston2022-02-28.110406.pkl --osm_name "Shubert Theatre" --yelp_name "Boch Center - Shubert Theatre" --new_value 2'

Attributes
----------
osm_name : str
    the osm name of the POI pair that should be edited
yelp_name : str
    the yelp name of the POI pair that should be edited
new_value : int
    the new value for the label
df : dataframe
    the pickled dataframe (.pkl-file) that contains the pair of POIs that should be edited
"""

# option to show all rows in the dataframe when printing
pd.set_option("display.max_rows", None, "display.max_columns", None)

def edit_label(osm_name, yelp_name, new_value, df):
    """
    Iterates through the dataframe to find the pair of POIs that should be updated and edits the label to the new value.

    Parameters
    ----------
    osm_name : str
        the osm name of the POI pair that should be edited
    yelp_name : str
        the yelp name of the POI pair that should be edited
    new_value : int
        the new value for the label. 0 = no match, 1 = match, 2 = unclear, 3 = irrelevant data, to be dropped.
    df : dataframe
        the pickled dataframe that contains the pair of POIs that should be edited

    """
    for index, pair in df.iterrows():
        if pair['osm_name'] == osm_name and pair['yelp_name'] == yelp_name:
            df.at[index,'match']=new_value
            #print('updated', pair)

def relabel(df):
    print(df.shape[0])
    for index, poi in df.iterrows(): #iterate through POIs in OSM dataset
        #if poi['match'] == 0:
        if 'Oakridge-41st' in poi['osm_name']:
            print("0: ", poi['osm_name'], " , ", poi['yelp_name'])
            df = df.drop(index)
    #print(df.shape[0])
        
                #print("0: ", poi['osm_name'], " , ", poi['yelp_name']) #Print pair for user to consider)
    #         #restrict only compare POIs if they are closely located
    #         if (lat_yelp < lat_osm+distance and lat_yelp > lat_osm-distance) and (lon_yelp < lon_osm+distance and lon_yelp > lon_osm-distance):     
    #             alreadyLabelled = False
    #             #check if pair already exist in labelled data:
    #             for i, labelled_poi in df_pairs.iterrows():
    #                 if labelled_poi['osm_name'] == poi_osm['name'] and labelled_poi['yelp_name'] == poi_yelp['name']:
    #                     alreadyLabelled = True
    #                     print('already labelled: ', labelled_poi['osm_name'], poi_osm['name'], labelled_poi['yelp_name'], poi_yelp['name'])
    #                     break
                        
    #             if not alreadyLabelled:        
    #                 poi_dist = distance_meters(lat_osm, lon_osm, lat_yelp, lon_yelp) #calculate distance between POIs  

    #                 print(poi_osm['name'], " , ", poi_yelp['name'], ' distance between: ', poi_dist) #Print pair for user to consider
    #                 if poi_osm['name'].lower() == poi_yelp['name'].lower(): #if names are an exact match
    #                     #print("exact")
    #                     print("1")
    #                     df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 1}, ignore_index=True)
    #                 elif levenshtein_similarity(poi_osm['name'].lower(), poi_yelp['name'].lower()) < 0.2:
    #                     df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 0}, ignore_index=True)
    #                     print("0")
    #                     #print("not similar ") #, poi_osm['name'], " , ", poi_yelp['name'], ' distance between: ', poi_dist)
    #                 else:
    #                     #print(poi_osm['name'], " , ", poi_yelp['name'], ' distance between: ', poi_dist)
    #                     #print("")
    #                     #num = 0
    #                     while True:
    #                         str_num = input() #Take input from user
    #                         try:
    #                             num = int(str_num)
    #                             if num not in range(0,4): #restrict input to 0/1
    #                                 print("Only numbers 0, 1, 2 and 3 valid")
    #                             else:
    #                                 break
    #                         except ValueError:
    #                             print("not int, try again")

    #                     #Add pair to DataFrame:
    #                     if num == 1:
    #                         df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 1}, ignore_index=True)
    #                     elif num == 0:
    #                         df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 0}, ignore_index=True)
    #                     elif num == 2:
    #                         df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 2}, ignore_index=True)
    #                     elif num == 3:
    #                         df_pairs = df_pairs.append({'osm_name': poi_osm['name'], 'yelp_name': poi_yelp['name'], 'osm_latitude': poi_osm['latitude'], 'osm_longitude': poi_osm['longitude'], 'yelp_latitude': poi_yelp['latitude'], 'yelp_longitude': poi_yelp['longitude'], 'distance': poi_dist, 'match': 3}, ignore_index=True)

    #df.to_pickle('temp.pkl') # save dataframe to pickle
    return df

def main():
    # parsing input arguments from command line to variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--osm_name', dest='osm_name')
    parser.add_argument('--yelp_name', dest='yelp_name')
    parser.add_argument('--new_value', dest='new_value')
    parser.add_argument('--df', dest = 'df')
    args = parser.parse_args()

    # reads the dataframe from the pickled file
    df = pd.read_pickle(args.df)
    #df = relabel(df)

    # reads the argument for the new label and edits the chosen label in the dataframe
    if int(args.new_value) == 1:
        edit_label(args.osm_name, args.yelp_name, 1, df)
    elif int(args.new_value) == 0:
        edit_label(args.osm_name, args.yelp_name, 0, df)
    elif int(args.new_value) == 2:
        edit_label(args.osm_name, args.yelp_name, 2, df)
    elif int(args.new_value) == 3:
        edit_label(args.osm_name, args.yelp_name, 3, df)

    #print(df)
    #df.to_pickle(args.df) # overwrites and saves dataframe

if __name__ == "__main__":
    main()