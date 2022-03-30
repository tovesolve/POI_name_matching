from json import load
from word_embeddings import *
from drop_label import *
from evaluation_metrics import *
from sentence_transformers import SentenceTransformer
from tokenizer import *

def load_df():
    df1 = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df2 = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')  
    df3 = pd.read_pickle('v0_df_pairs_vancouver_all2022-03-28.115404.pkl')
    df4 = pd.read_pickle('v0_df_pairs_vancouver_schools_libraries_community2022-03-25.153749.pkl') 
    df5 = pd.read_pickle('v0_df_pairs_nc2022-03-25.152112.pkl') 
    df = pd.concat([df1, df2, df3, df4, df5])
    df = drop_rows_with_label(df, 3)
    df = drop_rows_with_label(df, 2)
    #df = drop_exact_rows(df)
    return df

def calc_sBERT(df):
    model_sBERT = SentenceTransformer("all-mpnet-base-v2")
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        emb1 = sBERT(concat_token_list(tokenize_name(pair['osm_name'])), model_sBERT)
        emb2 = sBERT(concat_token_list(tokenize_name(pair['yelp_name'])), model_sBERT)
        score = sklearn_cosine_similarity(emb1, emb2)
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores

def calc_BERT(df):
    tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    model_BERT = BertModel.from_pretrained("bert-base-uncased")
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        emb1 = BERT(pair['osm_name'], model_BERT, tokenizer_BERT)
        emb2 = BERT(pair['yelp_name'], model_BERT, tokenizer_BERT)
        score = sklearn_cosine_similarity(emb1, emb2)
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores


def calc_BPEmb(df):
    dim=300
    vs=50000
    model_BPEmb = BPEmb(lang="en", dim=dim, vs=vs) 
    
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        
        emb1_BPEmb = BPEmb_embedding(concat_token_list(tokenize_name(pair['osm_name'])), model_BPEmb)
        emb2_BPEmb = BPEmb_embedding(concat_token_list(tokenize_name(pair['yelp_name'])), model_BPEmb)
        score = sklearn_cosine_similarity(emb1_BPEmb, emb2_BPEmb)
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores
    

def word_embedding_cosine_script(df, threshold, word_embedding):
    df_scores = word_embedding(df)
    dim=300
    vs=50000
    model_BPEmb = BPEmb(lang="en", dim=dim, vs=vs) 
    
    print("=========================False positives:========================================")
    for index, pair in df_scores.iterrows():
        if (pair['match'] is 0) and pair['score'] >= threshold:
            print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
            #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))

    print("==========================Flase negatives:========================================")
    for index, pair in df_scores.iterrows():
        if (pair['match'] is 1) and pair['score'] <= threshold:
            print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
            #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))    
    
    # print("==========================True positives:========================================")
    # for index, pair in df_scores.iterrows():
    #     if (pair['match'] == 1) and pair['score'] >= threshold:
    #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
    #         #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))    
   
    df_scores = classify_scores(df_scores, threshold)
    precision, recall, f1_score, matthew_correlation_coefficient = get_metrics(df_scores)
    print("threshold: ", threshold, " word embedding: ", word_embedding, " f1: ", f1_score, " precision: ", precision, " recall: ", recall, " matthew: ", matthew_correlation_coefficient)

def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    df = load_df()
    threshold = 0.6
    #embedding = calc_BPEmb
    #embedding = calc_sBERT
    embedding = calc_BERT
    word_embedding_cosine_script(df, threshold, embedding)

if __name__ == "__main__":
    main()
