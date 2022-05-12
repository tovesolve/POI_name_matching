from character_based_func import *
from drop_label import drop_rows_with_label, drop_exact_rows
from token_based_func import *
from token_based_func import jaccard_similarity, cosine_similarity
from evaluation_metrics import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse as a

"""
A baseline script to evaluate pure similarity functions. The script reads a dataframe, a list of similarity functions and a list of threshold values. 
The similarity functions are run seperately on the dataframe for the different thresholds. A bar-chart with evalaution metrics is generated.

Run script using: 'python -W ignore baseline.py --sim_funcs {similarity functions} --thresholds {float values} --metric {string metric}'
Example: 'python -W ignore baseline.py --sim_funcs cosine_similarity, levenshtein_similarity --thresholds 0.5 0.7 1.0 --metric f1_score'

Attributes
----------
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
    
    for threshold in thresholds:
        scores = []     
        for sim_func in sim_funcs:       
            df_scores = calculate_similarity_score(df, sim_func)
            df_scores = classify_scores(df_scores, threshold)
            precision, recall, f1_score, matthew_correlation_coefficient = get_metrics(df_scores)
            if metric == "precision":
                scores.append(precision)
            elif metric == "recall":
                scores.append(recall)
            elif metric == "f1_score":
                scores.append(f1_score)
            elif metric == "matthew":
                scores.append(matthew_correlation_coefficient)   
            print("threshold: ", threshold, " similarity func: ", sim_func, " f1: ", f1_score, " precision: ", precision, " recall: ", recall, " matthew: ", matthew_correlation_coefficient)
        
            # print("=========================False positives:========================================")
            # for index, pair in df_scores.iterrows():
            #     if (pair['match'] is 0) and pair['score'] >= threshold:
            #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'], pair['distance'])
            #         #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))

            # print("==========================False negatives:========================================")
            # for index, pair in df_scores.iterrows():
            #     if (pair['match'] is 1) and pair['score'] <= threshold:
            #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
            #         #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))    
            
            # print("==========================True positives:========================================")
            # for index, pair in df_scores.iterrows():
            #     if (pair['match'] == 1) and pair['score'] >= threshold:
            #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
            #         #print("tokenized to: ", tokenize(pair['osm_name']), " and: ", tokenize(pair['yelp_name']))

                  
        
        
        
        dict[threshold] = scores
    plot_evaluation_graph_sim_funcs(dict, thresholds, sim_funcs, metric)

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

def load_df():
    df1 = pd.read_pickle('v0.5_df_pairs_florida2022-02-28.094015.pkl')
    df2 = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')  
    df3 = pd.read_pickle('v0_df_pairs_vancouver_all2022-03-28.115404.pkl')
    df4 = pd.read_pickle('v0_df_pairs_vancouver_schools_libraries_community2022-03-25.153749.pkl') 
    df5 = pd.read_pickle('v0.5_df_pairs_nc2022-03-25.152112.pkl') 
    df = pd.concat([df1, df2, df3, df4, df5])
    df = drop_rows_with_label(df, 3)
    df = drop_rows_with_label(df, 2)
    #df = drop_exact_rows(df)
    return df

def main():
    
    # parsing input arguments from command line to variables
    parser = a.ArgumentParser()
    parser.add_argument('--sim_funcs', dest = 'sim_funcs', nargs="*", default=[])
    parser.add_argument('--thresholds', dest = 'thresholds', nargs="*", type=float, default=[])
    parser.add_argument('--metric', dest = 'metric')
    args = parser.parse_args()

    df = load_df()
    
    #iterates through the input similarity function list and adds as functions.
    sim_func_list = []
    for sim_func in args.sim_funcs:
        if sim_func == 'cosine':
            sim_func_list.append(cosine_similarity)
        elif sim_func == 'jaccard':
            sim_func_list.append(jaccard_similarity)
        elif sim_func == 'jaro':
            sim_func_list.append(jaro_similarity)
        elif sim_func == 'jaro_winkler':
            sim_func_list.append(jaro_winkler_similarity)
        elif sim_func == 'levenshtein':
            sim_func_list.append(levenshtein_similarity)

    #runs the baseline script with input arguments.
    baseline_script(df, sim_funcs=sim_func_list, thresholds=args.thresholds, metric=args.metric)

if __name__ == "__main__":
    main()
