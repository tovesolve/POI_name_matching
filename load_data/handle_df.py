import argparse
from itertools import count
from numpy import true_divide
import pandas as pd
from sqlite3 import Timestamp
import time

parser = argparse.ArgumentParser()

parser.add_argument('--osm_name', dest='osm_name')
parser.add_argument('--yelp_name', dest='yelp_name')
parser.add_argument('--new_value', dest='new_value')
parser.add_argument('--df', dest = 'df')
parser.add_argument('--class', dest = 'c')
args = parser.parse_args()

# Run script with name of both pois and new label (True/False)

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

'''
Counts and returns the number of each class together with a list of the true pairs (class1)
'''
def count_class(df):
    count_0 = 0
    count_1 = 0
    count_2 = 0
    true_pairs = []
    max_dist = 0
    sum_dist = 0
    for index, pair in df.iterrows():
        ind = int(pair['match'])
        if  ind == 0:
            count_0 +=1
        elif ind == 1:
            count_1 +=1
            true_pairs.append((pair['osm_name'], pair['yelp_name'], pair['distance']))
            if float(pair['distance']) > max_dist:
                max_dist = float(pair['distance'])
            sum_dist += float(pair['distance'])
        elif ind == 2:
            count_2 += 1
    
    #for pair in true_pairs:
    #    print(pair)
    
    print('max_dist', max_dist)
    print('mean_dist ', sum_dist/count_1)

    return count_0, count_1, count_2, true_pairs


# Wierd
def extra_tags(true_pair_list):
    df_osm = pd.read_pickle('df_osm_florida2022-02-25.094449.pkl')
    df_yelp = pd.read_pickle('df_yelp_florida2022-02-25.093315.pkl')
    for tuple in true_pair_list:
        for index, poi in df_osm.iterrows():
            if tuple[0] == poi['name']:
                print(tuple, poi['tags'])
                



def main():
    df = pd.read_pickle(args.df)
    print('number of pairs: ', df.shape[0])
    count_0, count_1, count_2, true_pairs = count_class(df)
    print('Number class 0: ', count_0)
    print('Number class 1: ', count_1)
    print('Number class 2: ', count_2)
    for pair in true_pairs:
        print(pair)
    #extra_tags(true_pairs)
    


if __name__ == "__main__":
    main()