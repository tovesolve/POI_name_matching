from character_based_func import *
from drop_label import drop_rows_with_label
from token_based_func import build_vector
from token_based_func import jaccard_similarity, cosine_similarity
from evaluation_metrics import *
from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaro_similarity as jaro
from nltk.metrics.distance import jaro_winkler_similarity as jaro_wrinkler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Baseline
# for list of sim_func:
    #for thresholds in threshold_list:
        #calculate sim_score (vi har nu y_predicted och y_truevalues)
        #calculate get_metrics
        #add f1 to a object together with sim_func and threshold
# plotta graf utifrån lista med objekt
def baseline_script(df, sim_func_list, threshold_list):
    dict = {}
    for sim_func in sim_func_list:
        f1_scores = []            
        for threshold in threshold_list:
            df_scores = calculate_similarity_score(df, sim_func)
            df_scores = score_to_label(df_scores, threshold)
            precision, recall, f1_score, matthew = get_metrics(df_scores)
            f1_scores.append(f1_score)
            #print("threshold: ", threshold, " similarity func: ", sim_func, " f1: ", f1_score)
        dict[sim_func] = f1_scores
    #plot graph#
    f1_comparision_graph(dict, threshold_list, sim_func_list)

def calculate_similarity_score(df, sim_func):
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    if sim_func == cosine_similarity:
        X, vectorized_df = build_vector(df)
        for index, pair in df.iterrows():
            score = sim_func(pair['osm_name'], pair['yelp_name'], X, vectorized_df)
            df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    else:
        for index, pair in df.iterrows():
            score = sim_func(pair['osm_name'], pair['yelp_name'])
            df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores

def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')
    
    df = drop_rows_with_label(df, 2)
    #baseline_script(df, sim_func_list=[levenshtein_similarity, damarau_levenshtein_similarity, jaro_similarity], threshold_list=[0.5, 0.7, 1])
    baseline_script(df, sim_func_list=[cosine_similarity], threshold_list=[1])


if __name__ == "__main__":
    main()
