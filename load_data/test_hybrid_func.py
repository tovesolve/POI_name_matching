from threading import local
from traceback import print_tb
from drop_label import drop_rows_with_label, drop_exact_rows
from baseline import calculate_similarity_score
from drop_label import *
from token_based_func import *
from evaluation_metrics import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import py_stringmatching
from math import log
from math import sqrt
import collections
from character_based_func import jaro_winkler_similarity, levenshtein_similarity

#from py_stringmatching import utils
#from py_stringmatching.similarity_measure.jaro import Jaro
#from py_stringmatching.similarity_measure.levenshtein import Levenshtein
#from py_stringmatching.similarity_measure.hybrid_similarity_measure import \
#                                                    HybridSimilarityMeasure

def softTFIDF(df, secondary_func=levenshtein_similarity, secondary_threshold = 0.5):
    corpus_list = get_corpus_list_for_pystringmatching(df) #create corpus from dataframe

    #manual softTFIDF:
    document_frequency = {}
    if corpus_list != None:
        for document in corpus_list:
            for element in set(document):
                document_frequency[element] = (document_frequency.get(element, 0) + 1)
    #print("doc freq:", document_frequency)

    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs
    for index, pair in df.iterrows():
        score = calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, secondary_threshold, secondary_func, document_frequency)
        #score = calc_softTFIDF_for_pair("Park Avenue Pizza", "Park Ave Pizza", corpus_list, secondary_threshold, secondary_func, document_frequency)
        
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)

    return df_scores

def sim_check_for_empty(*args):
    if len(args[0]) == 0 or len(args[1]) == 0:
        return True

def sim_check_for_exact_match(*args):
    if args[0] == args[1]:
        return True

def calc_softTFIDF_for_pair(osm_name, yelp_name, corpus_list, threshold, secondary_func, document_frequency):
    tokenized_osm_name = tokenize(osm_name)
    tokenized_yelp_name = tokenize(yelp_name)
    tf_x, tf_y = collections.Counter(tokenized_osm_name), collections.Counter(tokenized_yelp_name)

    # if the strings match exactly return 1.0
    if sim_check_for_exact_match(tokenized_osm_name, tokenized_yelp_name):
        return 1.0

    # if one of the strings is empty return 0
    if sim_check_for_empty(tokenized_osm_name, tokenized_yelp_name):
        return 0.0

    # find unique elements in the input lists and their document frequency 
    local_df = {}
    for element in tf_x:
        local_df[element] = local_df.get(element, 0) + 1
    for element in tf_y:
        local_df[element] = local_df.get(element, 0) + 1

    # if corpus is not provided treat input string as corpus
    corpus_size = len(corpus_list)
    curr_df, corpus_size = (local_df, 2) if corpus_list is None else ((document_frequency, corpus_size))
    
    similarity_map = {} #create dictionary with similar words
    for term_x in tf_x:
        #print(term_x)
        max_score = 0.0
        for term_y in tf_y:
            #score = levenshtein_similarity(term_x, term_y)
            score = secondary_func(term_x, term_y)
            #print("levenshtein for ", term_x, " and ", term_y, ": ", score)
            # adding sim only if it is above threshold and
            # highest for this element
            if score >= threshold and score > max_score:
                similarity_map[term_x] = (term_x, term_y, score)
                max_score = score
    
    first_string_pos = 0
    second_string_pos = 1
    sim_score_pos = 2

    result, v_x_2, v_y_2 = 0.0, 0.0, 0.0
    # soft-tfidf calculation
    for element in local_df.keys():
        if curr_df.get(element) is None:
            continue
        if element in similarity_map:
            sim = similarity_map[element]
            #print("sim", sim)
            idf_first = corpus_size / curr_df.get(sim[first_string_pos], 1) # size / hur många gånger den förekommer i corpus
            #print("idf first ", idf_first)
            idf_second = corpus_size / curr_df.get(sim[second_string_pos], 1)
            #print("idf second", idf_second)
            v_x = idf_first * tf_x.get(sim[first_string_pos], 0)
            v_y = idf_second * tf_y.get(sim[second_string_pos], 0)
            
            #v_x =  (log(idf_first) * log(tf_x.get(sim[first_string_pos], 0) + 1)) if True else (idf_first * tf_x.get(sim[first_string_pos], 0))
            #v_y =  (log(idf_second) * log(tf_y.get(sim[second_string_pos], 0) + 1)) if True else (tf_y.get(sim[second_string_pos], 0))
            #print("v_x:", v_x)
            #print("v_y:", v_y)
            result += v_x * v_y * sim[sim_score_pos]
            #print("result: ", result)
        # denominator
        idf = corpus_size / curr_df[element]
        #print("idf", idf)
        v_x = idf * tf_x.get(element, 0)    
        #v_x = (log(idf) * log(tf_x.get(element, 0)  + 1)) if True else (idf * tf_x.get(element, 0) )
        v_x_2 += v_x * v_x
        #v_y = (log(idf) * log(tf_x.get(element, 0)  + 1)) if True else (idf * tf_x.get(element, 0) )
        v_y = idf * tf_y.get(element, 0)
        #print("vy", v_y)
        v_y_2 += v_y * v_y
        #print("vx2", v_x_2)
        #print("vy2", v_y_2)
    #print("result soft-tfidf for ", osm_name, " and ", yelp_name, ": ")
    score = result if (v_x_2 == 0 or v_y_2 == 0)  else result / (sqrt(v_x_2) * sqrt(v_y_2))
    #print(score)
    return score

def calc_softTFIDF_for_pair_package(osm_name, yelp_name, corpus_list, threshold, secondary_func, document_frequency):
    tokenized_osm_name = tokenize(osm_name)
    tokenized_yelp_name = tokenize(yelp_name)
    soft_tfidf = py_stringmatching.SoftTfIdf(corpus_list, sim_func=secondary_func, threshold=threshold)
    package_score = soft_tfidf.get_raw_score(tokenized_osm_name, tokenized_yelp_name)
    return package_score


def TFIDF(df):
    corpus_list = get_corpus_list_for_pystringmatching(df) #create corpus from dataframe

    #manual TFIDF:
    document_frequency = {}
    if corpus_list != None:
        for document in corpus_list:
            for element in set(document):
                document_frequency[element] = (document_frequency.get(element, 0) + 1)
    #print("doc freq:", document_frequency)

    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs
    for index, pair in df.iterrows():
        score = calc_tfidf_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, document_frequency)
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)

    return df_scores


def calc_tfidf_for_pair(osm_name, yelp_name, corpus_list, document_frequency):
    tokenized_osm_name = tokenize(osm_name)
    tokenized_yelp_name = tokenize(yelp_name)
    tf_x, tf_y = collections.Counter(tokenized_osm_name), collections.Counter(tokenized_yelp_name)

    # if the strings match exactly return 1.0
    if sim_check_for_exact_match(tokenized_osm_name, tokenized_yelp_name):
        return 1.0

    # if one of the strings is empty return 0
    if sim_check_for_empty(tokenized_osm_name, tokenized_yelp_name):
        return 0.0

    # find unique elements in the input lists and their document frequency 
    local_df = {}
    for element in tf_x:
        local_df[element] = local_df.get(element, 0) + 1
    for element in tf_y:
        local_df[element] = local_df.get(element, 0) + 1

    # # if corpus is not provided treat input string as corpus
    corpus_size = len(corpus_list)
    curr_df, corpus_size = (local_df, 2) if corpus_list is None else ((document_frequency, corpus_size))
   
    #manual tfidf:    
    idf_element, v_x, v_y, v_x_y, v_x_2, v_y_2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # tfidf calculation
    for element in local_df.keys():
        df_element = curr_df.get(element)
        if df_element is None:
            continue
        idf_element = corpus_size * 1.0 / df_element
        v_x = 0 if element not in tf_x else (log(idf_element) * log(tf_x[element] + 1)) if True else (
                idf_element * tf_x[element])
        v_y = 0 if element not in tf_y else (log(idf_element) * log(tf_y[element] + 1)) if True else (
                idf_element * tf_y[element])
        v_x_y += v_x * v_y
        #print(element)
        #print("vx: ", v_x)
        v_x_2 += v_x * v_x
        #print(v_y)
        v_y_2 += v_y * v_y
    print("result manual tfidf:")
    score = (0.0 if v_x_y == 0 else v_x_y / (sqrt(v_x_2) * sqrt(v_y_2)))   
    print(score)
    return score
                    

def calc_TFIDF_for_pair_package(osm_name, yelp_name, corpus_list, document_frequency):
    tokenized_osm_name = tokenize(osm_name)
    tokenized_yelp_name = tokenize(yelp_name)
    tfidf = py_stringmatching.TfIdf(corpus_list, dampen=True)
    package_score = tfidf.get_sim_score(tokenized_osm_name, tokenized_yelp_name)
    print("result tfidf from package:")
    print(package_score)
    return package_score


def tfidf_script(df, sim_funcs, primary_thresholds, secondary_thresholds, metric):
    dict = {}
    for sim_func in sim_funcs:
        scores = []           
        for primary_threshold in primary_thresholds:
            for secondary_threshold in secondary_thresholds:
                df_scores = softTFIDF(df, secondary_func=sim_func, secondary_threshold = secondary_threshold)
                #df_scores = TFIDF(df)

                print("=========================False positives:========================================")
                for index, pair in df_scores.iterrows():
                    if (pair['match'] is 0) and pair['score'] >= primary_threshold:
                        print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
                        print("tokenized to: ", tokenize(pair['osm_name']), " and: ", tokenize(pair['yelp_name']))

                print("==========================Flase negatives:========================================")
                for index, pair in df_scores.iterrows():
                    if (pair['match'] is 1) and pair['score'] <= primary_threshold:
                        print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
                        print("tokenized to: ", tokenize(pair['osm_name']), " and: ", tokenize(pair['yelp_name']))

                print("==========================True positives:========================================")
                for index, pair in df_scores.iterrows():
                    if (pair['match'] == 1) and pair['score'] >= primary_threshold:
                        print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
                        print("tokenized to: ", tokenize(pair['osm_name']), " and: ", tokenize(pair['yelp_name']))



                df_scores = classify_scores(df_scores, primary_threshold)
                precision, recall, f1_score, matthew_correlation_coefficient = get_metrics(df_scores)
                if metric == "precision":
                    scores.append(precision)
                elif metric == "recall":
                    scores.append(recall)
                elif metric == "f1_score":
                    scores.append(f1_score)
                elif metric == "matthew":
                    scores.append(matthew_correlation_coefficient)   
                print("primary_threshold: ", primary_threshold, " similarity func: ", sim_func, " f1: ", f1_score, " precision: ", precision, " recall: ", recall, " matthew: ", matthew_correlation_coefficient)
        dict[sim_func] = scores
    
    print(dict)
    threshold_tuples = [] 
    for i in range(len(primary_thresholds)):
        for j in range(len(secondary_thresholds)):
            threshold_tuples.append((primary_thresholds[i], secondary_thresholds[j]))
    print(threshold_tuples)

    plot_evaluation_graph(dict, threshold_tuples, sim_funcs, metric)


def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    df1 = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df2 = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')  
    df3 = pd.read_pickle('v0_df_pairs_vancouver_all2022-03-28.115404.pkl')
    df4 = pd.read_pickle('v0_df_pairs_vancouver_schools_libraries_community2022-03-25.153749.pkl') 
    df5 = pd.read_pickle('v0_df_pairs_nc2022-03-25.152112.pkl') 
    df = pd.concat([df1, df2, df3, df4, df5])
    df = drop_rows_with_label(df, 3)
    df = drop_rows_with_label(df, 2)
    #df = drop_exact_rows(df)
    tfidf_script(df, [jaro_winkler_similarity], [0.42],[0.85], 'f1_score')
    # df_with_scores = softTFIDF(df, secondary_func=jaro_winkler_similarity, secondary_threshold=0.8)
    # #df_with_scores = TFIDF(df, secondary_func=jaro_winkler_similarity, secondary_threshold=0.8)
    
    # # for index, pair in df_with_scores.iterrows():
    # #     if (pair['match'] is 0) and pair['score'] > 0.6:
    # #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
    
    # df_with_classes = classify_scores(df_with_scores, threshold=0.6)
    # # for index, pair in df_with_classes.iterrows():
    # #     if pair['match'] is not pair['score']:
    # #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
    
    # precision, recall, f1_score, matthew_correlation_coefficient = get_metrics(df_with_classes)
    # print("precision: ", precision)
    # print("recall: ", recall)
    # print("f1-score: ", f1_score)
    # print("matthew-corr-coeff: ", matthew_correlation_coefficient)

if __name__ == "__main__":
    main()
























def old_main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    df1 = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df2 = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')    
    df = pd.concat([df1, df2])
    df = drop_rows_with_label(df, 2)
    corpus_list = get_corpus_list_for_pystringmatching(df)
    
    document_frequency = {}

    #manual softTFIDF:
    if corpus_list != None:
        for document in corpus_list:
            for element in set(document):
                document_frequency[element] = (document_frequency.get(element, 0) + 1)
    corpus_size = len(corpus_list)
    print("doc freq:", document_frequency)
    
    threshold = 0.4
    
    tf_x, tf_y = collections.Counter(['Park', 'Avenue', 'Pizza']), collections.Counter(['Park', 'Ave', 'Pizza'])
    print("tf_x ", tf_x)
    print("tf_y ", tf_y)
    # find unique elements in the input lists and their document frequency 
    local_df = {}
    for element in tf_x:
        local_df[element] = local_df.get(element, 0) + 1
    for element in tf_y:
        local_df[element] = local_df.get(element, 0) + 1
        
    # print("local", local_df)

    # # if corpus is not provided treat input string as corpus
    curr_df, corpus_size = (local_df, 2) if corpus_list is None else ((document_frequency, corpus_size))
    
    # print("curr", curr_df)
    # print("size", corpus_size) #number of documents (POIs) in corpus

    # # if corpus is not provided treat input string as corpus
    # curr_df, corpus_size = (local_df, 2)
    
    # similarity_map = {}
    # for term_x in tf_x:
    #     #print(term_x)
    #     max_score = 0.0
    #     for term_y in tf_y:
    #         #score = levenshtein_similarity(term_x, term_y)
    #         score = levenshtein_similarity(term_x, term_y)
    #         # adding sim only if it is above threshold and
    #         # highest for this element
    #         if score > threshold and score > max_score:
    #             similarity_map[term_x] = (term_x, term_y, score)
    #             max_score = score
    # print(similarity_map)
    
    # first_string_pos = 0
    # second_string_pos = 1
    # sim_score_pos = 2

    # result, v_x_2, v_y_2 = 0.0, 0.0, 0.0
    # # soft-tfidf calculation
    # for element in local_df.keys():
    #     if curr_df.get(element) is None:
    #         continue
    #     # numerator
    #     print("element", element)
    #     print(tf_x.get(element, 0))
    #     if element in similarity_map:
    #         sim = similarity_map[element]
    #         print("sim", sim)
    #         idf_first = corpus_size / curr_df.get(sim[first_string_pos], 1) # size / hur många gånger den förekommer i corpus
    #         print("idf first ", idf_first)
    #         idf_second = corpus_size / curr_df.get(sim[second_string_pos], 1)
    #         print("idf second", idf_second)
    #         v_x = idf_first * tf_x.get(sim[first_string_pos], 0)
    #         v_y = idf_second * tf_y.get(sim[second_string_pos], 0)
    #         print(v_x)
    #         result += v_x * v_y * sim[sim_score_pos]
    #         print(result)
    #     # denominator
    #     idf = corpus_size / curr_df[element]
    #     print("idf", idf)
    #     print("tf", tf_x.get(element, 0))
    #     v_x = idf * tf_x.get(element, 0)
    #     v_x_2 += v_x * v_x
    #     v_y = idf * tf_y.get(element, 0)
    #     print("tf_y", tf_y.get(element, 0))
    #     print("vy", v_y)
    #     v_y_2 += v_y * v_y
    #     print("vx2", v_x_2)
    #     print("vy2", v_y_2)
    # print("result soft:")
    # print(result if (v_x_2 == 0 or v_y_2 == 0)  else result / (sqrt(v_x_2) * sqrt(v_y_2)))
       
    #manual tfidf:    
    idf_element, v_x, v_y, v_x_y, v_x_2, v_y_2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # tfidf calculation
    for element in local_df.keys():
        df_element = curr_df.get(element)
        if df_element is None:
            continue
        idf_element = corpus_size * 1.0 / df_element
        v_x = 0 if element not in tf_x else (log(idf_element) * log(tf_x[element] + 1)) if True else (
                idf_element * tf_x[element])
        v_y = 0 if element not in tf_y else (log(idf_element) * log(tf_y[element] + 1)) if True else (
                idf_element * tf_y[element])
        v_x_y += v_x * v_y
        print(element)
        print("vx: ", v_x)
        v_x_2 += v_x * v_x
        print(v_y)
        v_y_2 += v_y * v_y
    print("result manual tfidf:")
    print(0.0 if v_x_y == 0 else v_x_y / (sqrt(v_x_2) * sqrt(v_y_2)))    
                    
    #soft tfidf using package:
    #soft_tfidf = py_stringmatching.SoftTfIdf(corpus_list, sim_func=levenshtein_similarity, threshold=threshold)
    #print(soft_tfidf.get_raw_score(['Park', 'Avenue', 'Pizza'], ['Park', 'Ave', 'Pizza']))
    
    #tfidf using package:
    #tfidf = py_stringmatching.TfIdf(corpus_list, dampen=True)
    #print(tfidf.get_sim_score(['Park', 'Avenue', 'Pizza'], ['Park', 'Ave', 'Pizza']))