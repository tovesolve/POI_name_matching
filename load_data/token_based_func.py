from tokenizer import tokenize_on_space
from nltk.metrics.distance import jaccard_distance as jaccard
import pandas as pd

def jaccard_similarity(str1, str2):
    tokens1 = tokenize_on_space(str1)
    tokens2 = tokenize_on_space(str2)
    j = jaccard(tokens1, tokens2)
    normalized_jaccard = 1-j
    return normalized_jaccard
    
def main():
    #import nltk
    #nltk.download('punkt')
    
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    
    j = jaccard_similarity('hello token', 'hello tok')
    print('jaccard: ', j)

if __name__ == "__main__":
    main()
