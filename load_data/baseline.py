from character_based_func import *
from drop_label import drop_rows_with_label
from token_based_func import *
from token_based_func import jaccard_similarity, cosine_similarity
from evaluation_metrics import *
from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaro_similarity as jaro
from nltk.metrics.distance import jaro_winkler_similarity as jaro_wrinkler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse as a

"""
A baseline script to evaluate pure similarity functions. The script reads a dataframe, a list of similarity functions and a list of threshold values. 
The similarity functions are run seperately on the dataframe for the different thresholds. A bar-chart with evalaution metrics is generated.

Run script using: 'python3 -W ignore baseline.py --df {pickled dataframe file} --sim-funcs {similarity functions} --thresholds {float values}'
Example: 'python3 -W ignore baseline.py --df v0_df_pairs_florida2022-02-28.094015.pkl --sim-funcs cosine_similarity, levenshtein_similarity --thresholds 0.5 0.7 1.0'

Attributes
----------
df : dataframe
    the pickled dataframe (.pkl-file) that contains the pair of POIs that should be evaluated
sim_funcs : list
    a list of similarity functions to be used in the evaluation
thresholds : list
    a list of float values to be used as thresholds in the evaluation
plot_metric : str
    the metric to be used in the generated plot
"""

# option to show all rows in the dataframe when printing
pd.set_option("display.max_rows", None, "display.max_columns", None)

def baseline_script(df, sim_funcs, thresholds, metric):
    """
    Iterates through the similarity functions in sim_funcs for each threshold in thresholds. Calculates precision, recall, f1_score and matthew correlation.
    Creates a dictionary {'similarity funcion: scores'} where scores is a list of the metric in the same order as the thresholds for each similarity function. 
    Plots a barchart over the given metric for the similarity functions and threshold.

    Parameters
    ----------
    df : dataframe
        the dataframe that contains the pair of POIs that should be evaluated
    sim_funcs : list
        a list of similarity functions to be used in the evaluation
    thresholds : list
        a list of float values to be used as thresholds in the evaluation
    metric : str
        the metric to be used in the generated plot
    """
    dict = {}
    for sim_func in sim_funcs:
        scores = []            
        for threshold in thresholds:
            df_scores = calculate_similarity_score(df, sim_func)
            df_scores = classify_scores(df_scores, threshold)
            precision, recall, f1_score, matthew = get_metrics(df_scores)
            if metric == "precision":
                scores.append(precision)
            elif metric == "recall":
                scores.append(recall)
            elif metric == "f1_score":
                scores.append(f1_score)
            elif metric == "matthew":
                scores.append(matthew)   
            #print("threshold: ", threshold, " similarity func: ", sim_func, " f1: ", f1_score)
        dict[sim_func] = scores
    plot_evaluation_graph(dict, thresholds, sim_funcs, metric)

def calculate_similarity_score(df, sim_func):
    """
    Iterrates through the dataframe and calculates the similarity score for each pair using the given sim_func. Adds column with the similarity score for each pair.
    If the similarity function is cosine_similarity, the vector is built based on the dataframe, before calculating the similarity score for each pair.

    Parameters
    ----------
    df : dataframe
        the dataframe that contains the pair of POIs of which the similarity score should be calculated.
    sim_func : list
        the similarity functions to be to calculate the similarity score for each pair
    """
    
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    if sim_func == cosine_similarity:
        X, vectorized_df = build_matrix(df)
        for index, pair in df.iterrows():
            score = sim_func(pair['osm_name'], pair['yelp_name'], X, vectorized_df)
            df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    else:
        for index, pair in df.iterrows():
            score = sim_func(pair['osm_name'], pair['yelp_name'])
            df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores

def main():
    
    # parsing input arguments from command line to variables
    parser = a.ArgumentParser()
    parser.add_argument('--dfs', dest = 'dfs', nargs="*", default=[])
    parser.add_argument('--sim_funcs', dest = 'sim_funcs', nargs="*", default=[])
    parser.add_argument('--thresholds', dest = 'thresholds', nargs="*", type=float, default=[])
    parser.add_argument('--metric', dest = 'metric')
    args = parser.parse_args()

    #iterates through the input dataframes and concatinates into one dataframe
    dfs_list = []
    for df in args.dfs:
        df = pd.read_pickle(df)
        dfs_list.append(df)
    df = pd.concat(dfs_list)
    
    # drops alla rows in the dataframe with label 2 (uncertain data points), to be excluded from the evaluation script.
    df = drop_rows_with_label(df, 2)
    
    #iterates through the input similarity function list and adds as functions.
    sim_func_list = []
    for sim_func in args.sim_funcs:
        if sim_func == 'cosine_similarity':
            sim_func_list.append(cosine_similarity)
        elif sim_func == 'jaccard_similarity':
            sim_func_list.append(jaccard_similarity)
        elif sim_func == 'jaro':
            sim_func_list.append(jaro)
        elif sim_func == 'jaro_wrinkler':
            sim_func_list.append(jaro_wrinkler)
        elif sim_func == 'levenshtein_similarity':
            sim_func_list.append(edit_distance)

    #runs the baseline script with input arguments.
    baseline_script(df, sim_funcs=sim_func_list, thresholds=args.thresholds, metric=args.metric)

if __name__ == "__main__":
    main()
