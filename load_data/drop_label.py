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

# parsing input arguments from command line to variables
parser = argparse.ArgumentParser()
parser.add_argument('--df', dest = 'df')
parser.add_argument('--label_value', dest='label_value')
args = parser.parse_args()

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

def main():
    # reads the dataframe from the pickled file
    df = pd.read_pickle(args.df)

    # dropa all rows with the given label
    df = drop_rows_with_label(df, int(args.label_value))

    # overwrites and saves dataframe
    df.to_pickle(args.df)

if __name__ == "__main__":
    main()