import argparse
import pandas as pd
from sqlite3 import Timestamp
import time

"""
A script used to drop all rows with a given label value, within a labelled dataframe from a pkl-file.
The script reads a current dataframe, drops all rows with the chosen label and overwrites the old dataframe and saves it as a pickled file with the same file name.

Run script using: 'python3 drop_label.py --df {pickled dataframe file} --label_value {int_value}'
Example: 'python3 drop_label.py --df df_pairs_boston2022-02-28.110406.pkl --label_value 3'

Attributes
----------
df : dataframe
    the pickled dataframe (.pkl-file) which label should be dropped
label_value : int
    the value of the label to be dropped
"""

# option to show all rows in the dataframe when printing
pd.set_option("display.max_rows", None, "display.max_columns", None)

def drop_rows_with_label(df, label_value):
    """
    Drops all rows in the dataframe with the label value

    Parameters
    ----------
    df : dataframe
        the pickled dataframe (.pkl-file) which label should be dropped
    label_value : int
        the value of the label to be dropped

    Returns
    -------
    dataframe
        an updated dataframe after dropping the rows
    """

    return df.drop(df[df['match'] == label_value].index)

def drop_exact_rows(df):
    """
    Drops all rows in the dataframe where osm_name==yelp_name, i.e. there is an exact match.

    Parameters
    ----------
    df : dataframe
        the pickled dataframe (.pkl-file) which label should be dropped

    Returns
    -------
    dataframe
        an updated dataframe after dropping the rows
    """
    print(df.shape)
    count = 0
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match']
    df_without_exact = pd.DataFrame(columns=data_colnames) #create dataframe whithout exact matches

    for index, pair in df.iterrows():
        if pair['osm_name'] == pair['yelp_name']:
            #print(pair)
            count +=1
        else:
            df_without_exact = df_without_exact.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match']}, ignore_index=True)


    print(count)
    print(df_without_exact.shape)

    return df_without_exact

def main():
    # parsing input arguments from command line to variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', dest = 'df')
    parser.add_argument('--label_value', dest='label_value')
    args = parser.parse_args()

    # reads the dataframe from the pickled file
    df = pd.read_pickle(args.df)

    # dropa all rows with the given label
    print(df.shape[0])
    df = drop_exact_rows(df)
    print(df)
    print(df.shape[0])
    # overwrites and saves dataframe
    #df.to_pickle(args.df)

if __name__ == "__main__":
    main()