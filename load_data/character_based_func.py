from evaluation_metrics import *
from nltk.metrics.distance import edit_distance as levenshtein
from nltk.metrics.distance import jaro_similarity as jaro
from nltk.metrics.distance import jaro_winkler_similarity as jaro_winkler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Character-based similarity functions used in the similarity evaluation scripts. 
"""

#NLTK package does not do to lower case

def levenshtein_similarity(str1, str2):
    """
    Calculates the levenshtein similarity between two strings and normalizes the similarity score.

    Parameters
    ----------
    str1 : str
        the first string to be compared
    str2 : str
        the second string to be compared

    Returns
    -------
    float
        the normalized similarity score
    """
    value = levenshtein(str1, str2, substitution_cost=1, transpositions=False)
    max_val = max(len(str1), len(str2))
    normalized_similarity = 1-(value/max_val)
    return normalized_similarity

#används inte
def damarau_levenshtein_similarity(str1, str2):
    """
    Calculates the damarau levenshtein similarity between two strings and normalizes the similarity score.
    
    The damarau levenshtein similarity calculates transpositions of characters as a single operation, which differs from levenshtein similarity.

    Parameters
    ----------
    str1 : str
        the first string to be compared
    str2 : str
        the second string to be compared

    Returns
    -------
    float
        the normalized similarity score
    """
    value = levenshtein(str1, str2, substitution_cost=1, transpositions=True)
    max_val = max(len(str1), len(str2))
    normalized_similarity = 1-(value/max_val)
    return normalized_similarity

def jaro_winkler_similarity(str1, str2):
    """
    Calculates the normalized jaro winkler similarity between two strings.
    
    The jaro winkler similarity adds a prefix character in front of the strings, for the first letter to weigh heigher, which differs from jaro similarity.

    Parameters
    ----------
    str1 : str
        the first string to be compared
    str2 : str
        the second string to be compared

    Returns
    -------
    float
        the normalized similarity score
    """
    return jaro_winkler(str1, str2)

def jaro_similarity(str1, str2):
    """
    Calculates the normalized jaro similarity between two strings.
    
    Parameters
    ----------
    str1 : str
        the first string to be compared
    str2 : str
        the second string to be compared

    Returns
    -------
    float
        the normalized similarity score
    """
    return jaro(str1, str2)
    
def main():
    #show all rows when printing dataframe
    pd.set_option("display.max_rows", None, "display.max_columns", None)

if __name__ == "__main__":
    main()
