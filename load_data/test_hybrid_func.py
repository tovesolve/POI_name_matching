from threading import local
from drop_label import drop_rows_with_label
from baseline import calculate_similarity_score
from token_based_func import *
from evaluation_metrics import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py_stringmatching
from math import log
from math import sqrt
import collections
from character_based_func import jaro_winkler_similarity, levenshtein_similarity

from py_stringmatching import utils
from py_stringmatching.similarity_measure.jaro import Jaro
from py_stringmatching.similarity_measure.levenshtein import Levenshtein
from py_stringmatching.similarity_measure.hybrid_similarity_measure import \
                                                    HybridSimilarityMeasure

def softTFIDF():
    return "todo"

def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
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
        
    print("local", local_df)

    # if corpus is not provided treat input string as corpus
    curr_df, corpus_size = (local_df, 2) if corpus_list is None else ((document_frequency, corpus_size))
    
    print("curr", curr_df)
    print("size", corpus_size) #number of documents (POIs) in corpus

    # if corpus is not provided treat input string as corpus
    curr_df, corpus_size = (local_df, 2)
    
    similarity_map = {}
    for term_x in tf_x:
        #print(term_x)
        max_score = 0.0
        for term_y in tf_y:
            #score = levenshtein_similarity(term_x, term_y)
            score = levenshtein_similarity(term_x, term_y)
            # adding sim only if it is above threshold and
            # highest for this element
            if score > threshold and score > max_score:
                similarity_map[term_x] = (term_x, term_y, score)
                max_score = score
    print(similarity_map)
    
    first_string_pos = 0
    second_string_pos = 1
    sim_score_pos = 2

    result, v_x_2, v_y_2 = 0.0, 0.0, 0.0
    # soft-tfidf calculation
    for element in local_df.keys():
        if curr_df.get(element) is None:
            continue
        # numerator
        print("element", element)
        print(tf_x.get(element, 0))
        if element in similarity_map:
            sim = similarity_map[element]
            print("sim", sim)
            idf_first = corpus_size / curr_df.get(sim[first_string_pos], 1) # size / hur många gånger den förekommer i corpus
            print("idf first ", idf_first)
            idf_second = corpus_size / curr_df.get(sim[second_string_pos], 1)
            print("idf second", idf_second)
            v_x = idf_first * tf_x.get(sim[first_string_pos], 0)
            v_y = idf_second * tf_y.get(sim[second_string_pos], 0)
            print(v_x)
            result += v_x * v_y * sim[sim_score_pos]
            print(result)
        # denominator
        idf = corpus_size / curr_df[element]
        print("idf", idf)
        print("tf", tf_x.get(element, 0))
        v_x = idf * tf_x.get(element, 0)
        v_x_2 += v_x * v_x
        v_y = idf * tf_y.get(element, 0)
        print("tf_y", tf_y.get(element, 0))
        print("vy", v_y)
        v_y_2 += v_y * v_y
        print("vx2", v_x_2)
        print("vy2", v_y_2)
        print("result soft:")
        print(result if (v_x_2 == 0 or v_y_2 == 0)  else result / (sqrt(v_x_2) * sqrt(v_y_2)))
       
    #manual tfidf:    
    #idf_element, v_x, v_y, v_x_y, v_x_2, v_y_2 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # # tfidf calculation
    # for element in local_df.keys():
    #     df_element = curr_df.get(element)
    #     if df_element is None:
    #         continue
    #     idf_element = corpus_size * 1.0 / df_element
    #     v_x = 0 if element not in tf_x else (log(idf_element) * log(tf_x[element] + 1)) if True else (
    #             idf_element * tf_x[element])
    #     v_y = 0 if element not in tf_y else (log(idf_element) * log(tf_y[element] + 1)) if True else (
    #             idf_element * tf_y[element])
    #     v_x_y += v_x * v_y
    #     print(element)
    #     print("vx: ", v_x)
    #     v_x_2 += v_x * v_x
    #     print(v_y)
    #     v_y_2 += v_y * v_y
    # print("result:")
    # print(0.0 if v_x_y == 0 else v_x_y / (sqrt(v_x_2) * sqrt(v_y_2)))    
                    
    #soft tfidf using package:
    soft_tfidf = py_stringmatching.SoftTfIdf(corpus_list, sim_func=levenshtein_similarity, threshold=threshold)
    print(soft_tfidf.get_raw_score(['Park', 'Avenue', 'Pizza'], ['Park', 'Ave', 'Pizza']))
    
    #tfidf using package:
    tfidf = py_stringmatching.TfIdf(corpus_list, dampen=True)
    print(tfidf.get_sim_score(['Park', 'Avenue', 'Pizza'], ['Park', 'Ave', 'Pizza']))

if __name__ == "__main__":
    main()