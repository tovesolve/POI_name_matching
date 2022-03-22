from encodings import normalize_encoding
from tokenizer import tokenize_on_space
from drop_label import drop_rows_with_label
from nltk.metrics.distance import jaccard_distance as jaccard
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
import numpy as np
import sys
import pandas as pd

"""
Token-based similarity functions used in the similarity evaluation scripts. 
"""

def get_corpus_from_df(df):
    """
    Creates a corpus from the given dataframe. The corpus is a set of the POIs name, meaning it doesnt include duplicates.

    Parameters
    ----------
    df : dataframe
        the dataframe used in the corpus
    Returns
    -------
    set
        a set containing all POI names in the dataframe.
    """
    corpus = []
    for index, row in df.iterrows():
        corpus.append(row['osm_name'])
        corpus.append(row['yelp_name'])
    return set(corpus)

def get_corpus_list_for_pystringmatching(df):
    """
    Creates a corpus from the given dataframe. The format is a 2D-array with seperators between words. Needed for package py-stringmatching.
    Example format: [["armandos", "pizzeria"], ["armandos", "pizza"]]
    To avoid duplicates we first transform the df into a set.

    Parameters
    ----------
    df : dataframe
        the dataframe used in the corpus
    Returns
    -------
    list
        a list containing all POI names in the dataframe in 2D-array format.
    """

    poi_set = get_corpus_from_df(df) #convert df to a set of POIs
    
    new_corpus = []
    for poi in poi_set:
        splitted_poi = poi.split() #split POIs on space-character.
        #print(osm_poi)
        new_corpus.append(splitted_poi)
    #print(new_corpus)
    return new_corpus


def count_vectorization(corpus):
    """
    Vectorizes the corpus using count vectorization and creates a document term matrix and a dataframe. These represent the frequency count of each token in the unique POI names in the corpus.

    Parameters
    ----------
    corpus : set
        the corpus unique POI names
    Returns
    -------
    matrix
        A document term matrix
    dataframe
        A dataframe of the document term matrix
    """
    
    vectorizer = CountVectorizer()
    
    # fit_transform creates a document term matrix from the corpus. Where every unique POI name is a row and every unique word in the corpus is a colum. The value at place ("POI name", "token") is the frequency of that token i the POI name, given by the countVectorization.
    # ex: Corpus: "Lucy's Restaurant", "Tove's Shop"
    # X:
    #                       Lucy's  Resturant   Tove's  Shop 
    #   Lucy's Restaurant    1          1         0      0
    #   TOve's Shop          0          0         1      1
    #
    # X at (0, 0) = 1
    # X at (1, 0) = 0
    X = vectorizer.fit_transform(corpus)
    
    # creates a dataframe from the X matrix
    tf_df = pd.DataFrame(X.toarray(), index=corpus, columns=vectorizer.get_feature_names_out())
    return X, tf_df

def tfidf_vectorization(corpus):
    """
    Vectorizes the corpus using tfidf vectorization and creates a document term matrix and a dataframe. These represent the tfidf value of each token in the unique POI names in the corpus.

    Parameters
    ----------
    corpus : set
        the corpus unique POI names
    Returns
    -------
    matrix
        A tfidf-weighted document term matrix
    dataframe
        A dataframe of the tfidf-weighted document term matrix
    """
      
    vectorizer = TfidfVectorizer()
    
    # fit_transform creates a tfidf-weighted document term matrix from the corpus. Where every unique POI name is a row and every unique word in the corpus is a colum. The value at place ("POI name", "token") is the tfidf-value of that token i the POI name, given by the tfidf-Vectorization.
    # ex: Corpus: "Lucy's Restaurant", "Tove's Shop"
    # X:
    #                       Lucy's  Resturant   Tove's  Shop 
    #   Lucy's Restaurant    1          1         0      0
    #   TOve's Shop          0          0         1      1
    #
    # X at (0, 0) = 1
    # X at (1, 0) = 0
    X = vectorizer.fit_transform(corpus)
    
    # creates a dataframe from the X matrix
    tfidf_df = pd.DataFrame(X.toarray(), index=corpus, columns=vectorizer.get_feature_names())
    return X, tfidf_df

def get_vector_from_name(poi_name, vectorized_df):
    """
    Returns the vectorization representation of the given poi_name from the vectorized_df dataframe.

    Parameters
    ----------
    poi_name : string
        the POI name
        
    vectorized_df : dataframe
        the dataframe containing the document term matrix
           
    Returns    
    -------
    series 
        the series representing a vectorized POI name in the document term matrix

    """
    return vectorized_df.loc[poi_name, :]

def jaccard_similarity(str1, str2, tokenizer_func=tokenize_on_space):
    """
    Tokenizes two strings using tokizer_func and calculates and normalizes the jaccard similarity between the token sets.

    Parameters
    ----------
    str1 : str
        the first string to be compared
    str2 : str
        the second string to be compared
    tokenizer_func : function
        the tokenizer function used to tokenize the strings into token sets

    Returns
    -------
    float
        the normalized similarity score
    """
    tokens1 = tokenizer_func(str1)
    tokens2 = tokenizer_func(str2)
    j = jaccard(tokens1, tokens2)
    normalized_jaccard = 1-j
    return normalized_jaccard

def cosine_similarity(str1, str2, matrix, vectorized_df):
    """
    Calculated the cosine_similarity between two strings, using their vectorized values from the document term matrix.

    Parameters
    ----------
    str1 : str
        the first string to be compared
    str2 : str
        the second string to be compared
    matrix : matrix
        the document term matrix with vectorized values
    vectorized_df : dataframe
        the dataframe from the document term matrix

    Returns
    -------
    float
        the normalized similarity score
    """
    v1 = get_vector_from_name(str1, vectorized_df)
    v2 = get_vector_from_name(str2, vectorized_df)    
    score = sklearn_cosine_similarity(np.array([v1]),np.array([v2]))
    return float(score[0])

def build_matrix(df):
    """
    Returns the document term matrix based on the given dataframe. The type of vectorization used is specified within the function.

    Parameters
    ----------
    df : dataframe
        the dataframe used to build the matrix.
           
    Returns
    -------
    matrix
        A document term matrix
    dataframe
        A dataframe of the document term matrix
    """
    
    df = drop_rows_with_label(df, 2)
    c = get_corpus_from_df(df)
    #X, vectorized_df = tfidf_vectorization(c)
    matrix, vectorized_df = count_vectorization(c)
    return matrix, vectorized_df
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    #df = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')

if __name__ == "__main__":
    main()
