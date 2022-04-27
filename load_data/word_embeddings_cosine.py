from json import load
from word_embeddings import *
from drop_label import *
from evaluation_metrics import *
from sentence_transformers import SentenceTransformer
from tokenizer import *

def load_df():
    df1 = pd.read_pickle('v0.5_df_pairs_florida2022-02-28.094015.pkl')
    df2 = pd.read_pickle('v0_df_pairs_boston2022-02-28.110406.pkl')
    df3 = pd.read_pickle('v0_df_pairs_vancouver_all2022-03-28.115404.pkl')
    df4 = pd.read_pickle('v0_df_pairs_vancouver_schools_libraries_community2022-03-25.153749.pkl')
    df5 = pd.read_pickle('v0.5_df_pairs_nc2022-03-25.152112.pkl')
    df = pd.concat([df1, df2, df3, df4, df5])
    df = drop_rows_with_label(df, 3)
    df = drop_rows_with_label(df, 2)
    #df = drop_exact_rows(df)
    return df

def sBERT(df):
    model_sBERT = SentenceTransformer("all-mpnet-base-v2")
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        emb1 = get_embedding_sBERT(pair['osm_name'], model_sBERT)
        emb2 = get_embedding_sBERT(pair['yelp_name'], model_sBERT)
        #print("sBERT embeddings:", emb1, emb2)
        score = sklearn_cosine_similarity(emb1, emb2)[0][0]
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores

def BERT(df):
    tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    model_BERT = BertModel.from_pretrained("bert-base-uncased")
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        emb1 = get_embedding_BERT(pair['osm_name'], model_BERT, tokenizer_BERT).tolist()
        emb2 = get_embedding_BERT(pair['yelp_name'], model_BERT, tokenizer_BERT).tolist()
        score = sklearn_cosine_similarity(emb1, emb2)[0][0]
        #print("embedding 1: ", emb1)

        #print(sklearn_cosine_similarity(emb1, emb2))
        #print(sklearn_cosine_similarity(emb1, emb2)[0][0])
        #print("semantic score for BERT ", pair['osm_name'], " and ", pair['yelp_name'], " :", score)
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores


def BPEmb(df):
    dim=300
    vs=50000
    model_BPEmb = BPEmbedding(lang="en", dim=dim, vs=vs) 
    
    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_scores = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        emb1_BPEmb = get_embedding_BPEmb(pair['osm_name'], model_BPEmb)
        emb2_BPEmb = get_embedding_BPEmb(pair['yelp_name'], model_BPEmb)
        #print("BPEmb embeddings: ", pair['osm_name'], emb1_BPEmb)
        score = sklearn_cosine_similarity(emb1_BPEmb, emb2_BPEmb)[0][0]
        #print("semantic score for BPEmb ", pair['osm_name'], " and ", pair['yelp_name'], " :", score)
        df_scores = df_scores.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': score}, ignore_index=True)
    return df_scores


def word_embedding_cosine_script(df, thresholds, embeddings_list, metric):
    # dim=300
    # vs=50000
    # model_BPEmb = BPEmb(lang="en", dim=dim, vs=vs)

    # print("=========================False positives:========================================")
    # for index, pair in df.iterrows():
    #     print(pair)
        #if (pair['match'] is 0) and pair['score'] >= threshold:
            #print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
            #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))

    #print("==========================False negatives:========================================")
    #for index, pair in df_scores.iterrows():
        #if (pair['match'] is 1) and pair['score'] <= threshold:
            #print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
            #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))

    # # print("==========================True positives:========================================")
    # # for index, pair in df_scores.iterrows():
    # #     if (pair['match'] == 1) and pair['score'] >= threshold:
    # #         print(pair['osm_name'], "    ", pair['yelp_name'], "    match: ", pair['match'], "  score: ", pair['score'])
    # #         #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))

    # df_scores = classify_scores(df_scores, threshold)
    # precision, recall, f1_score, matthew_correlation_coefficient = get_metrics(df_scores)

    dict = {}
    for threshold in thresholds:
        scores = []
        for word_embedding in embeddings_list:
            df_scores = word_embedding(df)
            data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
            df_scores_excel = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs

            # print("=========================False positives:========================================")
            # fp = 0
            # for index, pair in df_scores.iterrows():
            #     if (pair['match'] is 0) and pair['score'] >= threshold:
            #         fp = fp+1
            #         print(pair['osm_name'], "    ", pair['yelp_name'], " score: ", pair['score'])
            #         df_scores_excel = df_scores_excel.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': pair['score']}, ignore_index=True)

            #         #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))

            # print("==========================False negatives:========================================")
            # fn = 0
            # for index, pair in df_scores.iterrows():
            #     if (pair['match'] is 1) and pair['score'] <= threshold:
            #         fn = fn+1
            #         print(pair['osm_name'], "    ", pair['yelp_name'], "  score: ", pair['score'])
            #         df_scores_excel = df_scores_excel.append({'osm_name': pair['osm_name'], 'yelp_name': pair['yelp_name'], 'osm_latitude': pair['osm_latitude'], 'osm_longitude': pair['osm_longitude'], 'yelp_latitude': pair['yelp_latitude'], 'yelp_longitude': pair['yelp_longitude'], 'distance': pair['distance'], 'match': pair['match'], 'score': pair['score']}, ignore_index=True)
            #         #print("tokenized to: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['osm_name']))), " and: ", model_BPEmb.encode(concat_token_list(tokenize_name(pair['yelp_name']))))
            # print("FP: ", fp)
            # print("FN: ", fn)
            
            # with pd.ExcelWriter("BERT_cosine.xlsx") as writer:
            #     df_scores_excel.to_excel(writer)
            
            df_scores = classify_scores(df_scores, threshold)
            precision, recall, f1_score, matthew_correlation_coefficient = get_metrics(df_scores)
            if metric == "precision":
                scores.append(precision)
            elif metric == "recall":
                scores.append(recall)
            elif metric == "f1_score":
                scores.append(f1_score)
            elif metric == "matthew":
                scores.append(matthew_correlation_coefficient)
                
        dict[threshold] = scores
        print("word embedding: ", word_embedding.__name__, "threshold: ", threshold, " precision: ", precision, " recall: ", recall, " matthew: ", matthew_correlation_coefficient, " f1: ", f1_score)
    #plot_evaluation_graph_cosine_word_embeddings(dict, thresholds, embeddings_list, metric)

def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    df = load_df()
    metric = "f1_score"
    thresholds = [0.5, 0.6, 0.7, 0.8]
    embeddings_list = [BPEmb]
    #embedding = calc_BPEmb
    #embedding = calc_sBERT
    #print("BPEmb")
    #word_embedding_cosine_script(df, [0.6], [BPEmb], metric)
    #print("BERT")
    word_embedding_cosine_script(df, thresholds, embeddings_list, metric)

if __name__ == "__main__":
    main()
