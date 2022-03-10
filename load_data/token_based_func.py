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

def get_corpus_from_df(df):
    corpus = []
    for index, row in df.iterrows():
        corpus.append(row['osm_name'])
        corpus.append(row['yelp_name'])
    return set(corpus)

def tfidf_vectorization(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tfidf_df = pd.DataFrame(X.toarray(), index=corpus, columns=vectorizer.get_feature_names())
    return X, tfidf_df
    
def count_vectorization(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    tfidf_df = pd.DataFrame(X.toarray(), index=corpus, columns=vectorizer.get_feature_names())
    return X, tfidf_df

def get_vector_from_name(poi_name, vectorized_df):
    return vectorized_df.loc[poi_name, :]

def jaccard_similarity(str1, str2, tokenizer_func=tokenize_on_space):
    tokens1 = tokenizer_func(str1)
    tokens2 = tokenizer_func(str2)
    j = jaccard(tokens1, tokens2)
    normalized_jaccard = 1-j
    return normalized_jaccard

def cosine_similarity(str1, str2, X, vectorized_df):
    v1 = get_vector_from_name(str1, vectorized_df)
    v2 = get_vector_from_name(str2, vectorized_df)    
    score = sklearn_cosine_similarity(np.array([v1]),np.array([v2]))
    return float(score[0])

def build_vector(df):
    df = drop_rows_with_label(df, 2)
    c = get_corpus_from_df(df)
    X, vectorized_df = tfidf_vectorization(c)
    return X, vectorized_df
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    df = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')


if __name__ == "__main__":
    main()
