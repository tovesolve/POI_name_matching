"""
Shows characteristics of Dataset.


Functions
---------
    * count_classes

Run script:  python3 handle_df <args>
    args are defined below
Basic run:   python3 handle_df --df <file_name>
"""

import argparse
from itertools import count
from numpy import true_divide
import pandas as pd
from sqlite3 import Timestamp
import time

# Arguments that can be run with script
parser = argparse.ArgumentParser()
parser.add_argument('--osm_name', dest='osm_name')
parser.add_argument('--yelp_name', dest='yelp_name')
parser.add_argument('--new_value', dest='new_value')
parser.add_argument('--df', dest = 'df')
parser.add_argument('--class', dest = 'c')
args = parser.parse_args()

pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

def count_classes(df):
    '''
    Counts and returns the number of each class together with lists of pairs of class0, class 1, class2 and class3
    
    Parameters
    ----------
    df : pandas DataFrame
        Dataframe

    Returns
    -------
    count_0
        Number of pairs with class 0
    count_1
        Number of pairs with class 1
    count_2
        Number of pairs with class 2
    count_3
        Number of pairs with class 3
    class0_pairs
        list of pairs with class 0
    class1_pairs
        list of pairs with class 1
    class2_pairs
        list of pairs with class 2
    class3_pairs
        list of pairs with class 3
    '''
    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    class0_pairs = []
    class1_pairs = []
    class2_pairs = []
    class3_pairs = []
    max_dist = 0
    sum_dist = 0
    for index, pair in df.iterrows():
        ind = int(pair['match'])
        if  ind == 0:
            count_0 +=1
            class0_pairs.append((pair['osm_name'], pair['yelp_name'], pair['distance']))
        elif ind == 1:
            count_1 +=1
            class1_pairs.append((pair['osm_name'], pair['yelp_name'], pair['distance']))
            if float(pair['distance']) > max_dist:
                max_dist = float(pair['distance'])
            sum_dist += float(pair['distance'])
        elif ind == 2:
            count_2 += 1
            class2_pairs.append((pair['osm_name'], pair['yelp_name'], pair['distance']))
        elif ind == 3:
            count_3 += 1
            class3_pairs.append((pair['osm_name'], pair['yelp_name'], pair['distance']))

    print('max_dist', max_dist) # Print the maximum distance between to POIs with class 1
    print('mean_dist ', sum_dist/count_1) # Print the mean distance between POIs in the pairs

    return count_0, count_1, count_2, count_3, class0_pairs, class1_pairs, class2_pairs, class3_pairs


def extra_tags(true_pair_list):
    '''
    Prints tags from OSM POI from the true_pair_list
        
    Parameters
    ----------
    true_pair_list : list
        List of pairs that have class 1

    '''
    df_osm = pd.read_pickle('df_osm_florida2022-02-25.094449.pkl')
    df_yelp = pd.read_pickle('df_yelp_florida2022-02-25.093315.pkl')
    for tuple in true_pair_list:
        for index, poi in df_osm.iterrows():
            if tuple[0] == poi['name']:
                print(tuple, poi['tags'])
                


def similar(true_pairs):
    '''
    Creates and returns seperate lists for pairs where POI names are an exact match and where they differ.
    
    Parameters
    ----------
    true_pairs : list
        List of pairs that have class 1

    Returns
    -------
    exact_pairs
        list of pairs where the POIs have the exact same name
    similar_pairs
        list of pairs where the POIs have the similar names (are same point but name differs)
    '''
    exact_pairs = [] #Pairs where the POIs have the exact same name
    similar_pairs = [] #Pairs where the POIs are the same point but have different names

    for pair in true_pairs:
        if pair[0] == pair[1]:
            exact_pairs.append(pair)
        else:
            similar_pairs.append(pair)

    print("===Exact matches: =====")
    for pair in exact_pairs:
        print(pair)
    print("===Similar matches: =====")
    for pair in similar_pairs:
        print(pair)
    print("=============================================")
    print("Number exact matches: ", len(exact_pairs), "     Number similar matches: ", len(similar_pairs))
    return exact_pairs, similar_pairs



def main():
    df = pd.read_pickle(args.df) #Read dataframe from pickle

    print('number of pairs: ', df.shape[0])
    count_0, count_1, count_2, count_3, class0_pairs, class1_pairs, class2_pairs, class3_pairs = count_classes(df)
    print('Number class 0: ', count_0)
    print('Number class 1: ', count_1)
    print('Number class 2: ', count_2)
    print('Number class 3: ', count_3)
    # for pair in class2_pairs:
    #     print(pair)
    
    exact_pairs, similar_pairs = similar(class1_pairs)
    


if __name__ == "__main__":
    main()