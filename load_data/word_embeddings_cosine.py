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
    #BPEmb
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

    # #sBERT
    # # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # # model.save("all-mpnet-base-v2")
    # model_sBERT = SentenceTransformer("all-mpnet-base-v2")
    # emb1_sBERT = sBERT(poi1, model_sBERT)
    # emb2_sBERT = sBERT(poi2, model_sBERT)
    # print("Cosine similarity score using sBERT model: ", sklearn_cosine_similarity(emb1_sBERT, emb2_sBERT))
    
    # #BERT
    # tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    # model_BERT = BertModel.from_pretrained("bert-base-uncased")
    # emb1_BERT = BERT(poi1, model_BERT, tokenizer_BERT)
    # emb2_BERT = BERT(poi2, model_BERT, tokenizer_BERT)
    # print("Cosine similarity score using BERT model: ", sklearn_cosine_similarity(emb1_BERT, emb2_BERT))

    # #Word2vec
    # #model_word2vec = api.load("word2vec-google-news-300")  # load pre-trained word-vectors from gensim-data
    # #model_word2vec.save('vectors_word2vec.kv')
    # model_word2vec = KeyedVectors.load('vectors_word2vec.kv')
    # emb1_word2vec = word2vec_embedding(poi1, model_word2vec)
    # emb2_word2vec = word2vec_embedding(poi2, model_word2vec)
    # print("Cosine similarity score using Word2vec model: ", sklearn_cosine_similarity(emb1_word2vec.reshape(1,-1), emb2_word2vec.reshape(1,-1)))

    # #GloVe
    # #model_glove = api.load('glove-wiki-gigaword-200')  # load pre-trained word-vectors from gensim-data
    # #model_glove.save('vectors.kv')
    # model_glove = KeyedVectors.load('vectors.kv')
    # emb1_glove = glove_embedding(poi1, model_glove)
    # emb2_glove = glove_embedding(poi2, model_glove)
    # print("Cosine similarity score using glove model: ", sklearn_cosine_similarity(emb1_glove.reshape(1,-1), emb2_glove.reshape(1,-1)))

    # #Fasttext
    # #model_fasttext = api.load("fasttext-wiki-news-subwords-300")  # load pre-trained word-vectors from gensim-data
    # #model_fasttext.save('vectors_fasttext.kv')
    # model_fasttext = KeyedVectors.load('vectors_fasttext.kv')
    # emb1_fasttext = fasttext_embedding(poi1, model_fasttext)
    # emb2_fasttext = fasttext_embedding(poi2, model_fasttext)
    # print("Cosine similarity score using fasttext model: ", sklearn_cosine_similarity(emb1_fasttext.reshape(1,-1), emb2_fasttext.reshape(1,-1)))

if __name__ == "__main__":
    main()
