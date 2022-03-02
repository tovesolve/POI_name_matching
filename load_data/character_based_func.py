from drop_label import drop_rows_with_label
from evaluation_metrics import *
from nltk.metrics.distance import edit_distance
from nltk.metrics.distance import jaro_similarity as jaro
from nltk.metrics.distance import jaro_winkler_similarity as jaro_wrinkler
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def levenshtein_similarity(str1, str2):
    value = edit_distance(str1, str2, substitution_cost=1, transpositions=False)
    max_val = max(len(str1), len(str2))
    normalized_similarity = 1-(value/max_val)
    return normalized_similarity

def damarau_levenshtein_similarity(str1, str2):
    value = edit_distance(str1, str2, substitution_cost=1, transpositions=True)
    max_val = max(len(str1), len(str2))
    normalized_similarity = 1-(value/max_val)
    return normalized_similarity

def jaro_wrinkler_similarity(str1, str2):
    return jaro_wrinkler(str1, str2)

def jaro_similarity(str1, str2):
    return jaro(str1, str2)

def calculate_similarity_score(df, sim_func):
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    for index, pair in df.iterrows():
        score = sim_func(pair['osm_name'], pair['yelp_name'])
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df = pd.read_pickle('df_pairs_boston2022-02-28.110406.pkl')
    
    df = drop_rows_with_label(df, 2)
    df_scores_levenshtein = calculate_similarity_score(df, levenshtein_similarity)
    threshold = 0.7
    precision, recall, f1_score = get_metrics(df_scores_levenshtein, threshold)
    
    # av de vi hittade, hur många hade vi labelat till 1, dvs av de vi hittade, hur många skulle hittas?
    print("precision: ", precision)
    
    # hittade 1:0r / alla labelade 1:or, dvs hur stor andel av de vi VILLE hitta, hittade vi faktiskt?
    print("recall: ", recall)
    
    print("f1_score: ", f1_score)
    
    df_scores_dam = calculate_similarity_score(df, jaro_wrinkler_similarity)
    precision_dam, recall_dam, f1_score_dam = get_metrics(df_scores_dam, threshold)
    # av de vi hittade, hur många hade vi labelat till 1, dvs av de vi hittade, hur många skulle hittas?
    print("precision: ", precision_dam)
    
    # hittade 1:0r / alla labelade 1:or, dvs hur stor andel av de vi VILLE hitta, hittade vi faktiskt?
    print("recall: ", recall_dam)
    
    # 
    print("f1_score: ", f1_score_dam)

if __name__ == "__main__":
    main()
