import argparse
import pandas as pd
from sqlite3 import Timestamp
import time

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

# parsing input arguments from command line to variables
parser = argparse.ArgumentParser()
parser.add_argument('--osm_name', dest='osm_name')
parser.add_argument('--yelp_name', dest='yelp_name')
parser.add_argument('--new_value', dest='new_value')
parser.add_argument('--df', dest = 'df')
args = parser.parse_args()

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

def main():
    # reads the dataframe from the pickled file
    df = pd.read_pickle(args.df)

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
    df.to_pickle(args.df) # overwrites and saves dataframe

if __name__ == "__main__":
    main()