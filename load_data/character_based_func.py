from nltk.metrics.distance import edit_distance, jaro_winkler_similarity, jaro_similarity
import pandas as pd
import Levenshtein





def testNLTK(str1, str2):
    value = edit_distance(str1, str2, substitution_cost=1, transpositions=False)
    value_jaro_wrinkler = jaro_winkler_similarity(str1, str2)
    print(value_jaro_wrinkler)

def get_scores(df, sim_func, threshold):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    real_pos = 0
    real_neg = 0
    true_pos_pairs = []
    false_neg_pairs = []
    for index, pair in df.iterrows():
        score = sim_func(pair['osm_name'], pair['yelp_name'])
        if score > threshold:
            print(pair['osm_name'], ' ', pair['yelp_name'], ' score: ', score)
            if pair['match'] == True:
                real_pos +=1
                true_pos +=1
                true_pos_pairs.append(pair)
            else:
                real_neg +=1
                false_pos +=1
        else:
            if pair['match'] == True:
                false_neg +=1
                print('the false negative: ', pair['osm_name'], pair['yelp_name'], ' , score: ', score)
                real_pos +=1
                false_neg_pairs.append(pair)
            else:
                real_neg+=1
                true_neg +=1
    print('Similarity function: ', sim_func)
    print('Threshold: ', 0.9)
    print('True positives: ', true_pos)
    print('True negatives: ', true_neg)
    print('False positives: ', false_pos)
    print('False negatives: ', false_neg)
    print('real_pos: ', real_pos)
    print('real_neg: ', real_neg)
    print('Accuracy: ')
    print('Recall: ', true_pos/(true_pos + false_neg))  #Recall = TruePositives / (TruePositives + FalseNegatives)
    print('Precision: ', true_pos/(true_pos+false_pos)) #precision = TruePositives / (TruePositives + FalsePositives)
    print('')
    print('True positive pairs: ')
    print(true_pos_pairs)
    print('')
    print('False negative pairs: ')
    print(false_neg_pairs)
    print('=================')


def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    df = pd.read_pickle('df_pairs_2022-02-23.111124.pkl')
    #print(df)
    get_scores(df, Levenshtein.ratio, 0.9)
    get_scores(df, jaro_similarity, 0.9)
    get_scores(df, jaro_winkler_similarity, 0.9)
    print('Number of rows in df: ', df.shape[0])

if __name__ == "__main__":
    main()
