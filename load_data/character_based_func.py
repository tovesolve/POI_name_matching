from drop_label import drop_rows_with_label
from baseline import calculate_similarity_score
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
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')
    
    df = drop_rows_with_label(df, 2)
    df_scores_levenshtein = calculate_similarity_score(df, levenshtein_similarity)
    threshold = 0.7
    df_scores_levenshtein = score_to_label(df_scores_levenshtein, threshold)
    precision, recall, f1_score, matthew = get_metrics(df_scores_levenshtein)
    
    # av de vi hittade, hur många hade vi labelat till 1, dvs av de vi hittade, hur många skulle hittas?
    print("precision: ", precision)
    
    # hittade 1:0r / alla labelade 1:or, dvs hur stor andel av de vi VILLE hitta, hittade vi faktiskt?
    print("recall: ", recall)
    
    print("f1_score: ", f1_score)
    print("matthew: ", matthew)
    
    df_scores_dam = calculate_similarity_score(df, jaro_wrinkler_similarity)
    df_scores_dam = score_to_label(df_scores_dam, threshold)
    precision_dam, recall_dam, f1_score_dam, matthew_dam = get_metrics(df_scores_dam)
    # av de vi hittade, hur många hade vi labelat till 1, dvs av de vi hittade, hur många skulle hittas?
    print("precision: ", precision_dam)
    
    # hittade 1:0r / alla labelade 1:or, dvs hur stor andel av de vi VILLE hitta, hittade vi faktiskt?
    print("recall: ", recall_dam)
    
    # 
    print("f1_score: ", f1_score_dam)
    
    print("matthew: ", matthew_dam)

if __name__ == "__main__":
    main()
