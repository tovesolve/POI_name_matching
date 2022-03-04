from encodings import normalize_encoding
from tokenizer import tokenize_on_space
from nltk.metrics.distance import jaccard_distance as jaccard
from scipy import spatial
import pandas as pd

def jaccard_similarity(str1, str2, tokenizer_func=tokenize_on_space):
    tokens1 = tokenizer_func(str1)
    tokens2 = tokenizer_func(str2)
    j = jaccard(tokens1, tokens2)
    normalized_jaccard = 1-j
    return normalized_jaccard


def cosine_similarity(str1, str2, tokenizer_func=tokenize_on_space):
    tokens1 = tokenizer_func(str1)
    tokens2 = tokenizer_func(str2)
    l1 =[];l2 =[]
  
    # form a set containing keywords of both strings 
    union_vector = tokens1.union(tokens2) 
    for w in union_vector:
        if w in tokens1: l1.append(1) # create a vector
        else: l1.append(0)
        if w in tokens2: l2.append(1)
        else: l2.append(0)
    
    cosine = spatial.distance.cosine(l1, l2)
    normalized_cosine = 1 - cosine
    return normalized_cosine
    
def main():
    #import nltk
    #nltk.download('punkt')
    
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    
    j = jaccard_similarity('hello token', 'hello')
    print('jaccard: ', j)
    cosine = cosine_similarity('hello token', 'hello')
    print('cosine: ', cosine)

if __name__ == "__main__":
    main()
