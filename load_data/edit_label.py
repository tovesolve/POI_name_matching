import argparse
import pandas as pd
from sqlite3 import Timestamp
import time

parser = argparse.ArgumentParser()

parser.add_argument('--osm_name', dest='osm_name')
parser.add_argument('--yelp_name', dest='yelp_name')
parser.add_argument('--new_value', dest='new_value')
args = parser.parse_args()

# Run script with name of both pois and new label (True/False)

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe


'''
Label closely located pairs and save to dataframe.
'''
def edit_label(df, osm_name, yelp_name, new_value):
    for index, pair in df.iterrows():
        if pair['osm_name'] == osm_name and pair['yelp_name'] == yelp_name:
            df.at[index,'match']=new_value
            print('updated', pair)





def main():
    df = pd.read_pickle('df_pairs_1645521936.8028078.pkl')
    if args.new_value == 'True':
        edit_label(df, args.osm_name, args.yelp_name, True)
    elif args.new_value == 'False':
        edit_label(df, args.osm_name, args.yelp_name, False)
    print(df)


if __name__ == "__main__":
    main()