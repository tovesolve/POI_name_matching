import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from word_embeddings_cosine import *
from sklearn.model_selection import train_test_split
import character_based_func
import token_based_func
import test_hybrid_func
import semantic_soft_tfidf
from bpemb import BPEmb as BPEmbedding
import word_embeddings_cosine
import word_embeddings
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold

def prepareCorpus(df):
    corpus_list = token_based_func.get_corpus_list_for_pystringmatching(df) #create corpus from dataframe

    #manual TFIDF:
    document_frequency = {}
    if corpus_list != None:
        for document in corpus_list:
            for element in set(document):
                document_frequency[element] = (document_frequency.get(element, 0) + 1)
    
    return corpus_list, document_frequency

def createDataFrame(df):
    # data_colnames = ['levenshtein', 'jaro', 'jaro_winkler', 'cosine', 'jaccard', 'tfidf']
    # data_colnames = ['osm_name', 'yelp_name', 'softtfidf', 'semanticsofttfidf', 'sbert', 'bpemb', 'bert', 'match']

    data_colnames = ['osm_name', 'yelp_name', 'levenshtein', 'jaro', 'jaro_winkler', 'cosine', 'jaccard', 'tfidf', 'softtfidf', 'semanticsofttfidf', 'sbert', 'bpemb', 'bert', 'match']
    df_with_similarity_scores = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs
   
   
    X, vectorized_df = token_based_func.build_matrix(df)
    corpus_list, document_frequency = prepareCorpus(df)
    bpemb_model = BPEmbedding(lang="en", dim=300, vs=50000)
    sBERT_model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    BERT_model = BertModel.from_pretrained("bert-base-uncased")
    
    for index, pair in df.iterrows():
        levenshtein_score = character_based_func.levenshtein_similarity(pair['osm_name'], pair['yelp_name'])
        jaro_score = character_based_func.jaro_similarity(pair['osm_name'], pair['yelp_name'])
        jaro_winkler_score = character_based_func.jaro_winkler_similarity(pair['osm_name'], pair['yelp_name'])
        cosine_score = token_based_func.cosine_similarity(pair['osm_name'], pair['yelp_name'], X, vectorized_df)
        jaccard_score = token_based_func.jaccard_similarity(pair['osm_name'], pair['yelp_name']) #this is tokenized on space as default, can be changed
        tfidf_score = test_hybrid_func.calc_tfidf_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, document_frequency)
        softtfidf_score = test_hybrid_func.calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.9, character_based_func.jaro_winkler_similarity, document_frequency) #dessa thresholds är med det bästa vi testat hittils, ev uppdatera
        semanticsofttfidf_score = semantic_soft_tfidf.calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.85, character_based_func.jaro_winkler_similarity, 0.7, bpemb_model, document_frequency, tokenizer_BERT) #dessa thresholds är med det bästa vi testat hittils, ev uppdatera. (tokenizer BERT används inte)
        sbert_score = sklearn_cosine_similarity(word_embeddings.get_embedding_sBERT(pair['osm_name'], sBERT_model), word_embeddings.get_embedding_sBERT(pair['yelp_name'], sBERT_model))[0][0]
        bpemb_score = sklearn_cosine_similarity(word_embeddings.get_embedding_BPEmb(pair['osm_name'], bpemb_model), word_embeddings.get_embedding_BPEmb(pair['yelp_name'], bpemb_model))[0][0]
        bert_score = sklearn_cosine_similarity(word_embeddings.get_embedding_BERT(pair['osm_name'], BERT_model, tokenizer_BERT), word_embeddings.get_embedding_BERT(pair['yelp_name'], BERT_model, tokenizer_BERT))[0][0]
        # df_with_similarity_scores = df_with_similarity_scores.append({'levenshtein': levenshtein_score, 'jaro': jaro_score, 'jaro_winkler':jaro_winkler_score, 'cosine':cosine_score, 'jaccard':jaccard_score, 'tfidf':tfidf_score, 'match':pair['match']}, ignore_index=True) 
        # df_with_similarity_scores = df_with_similarity_scores.append({'osm_name':pair['osm_name'], 'yelp_name':pair['yelp_name'], 'softtfidf':softtfidf_score, 'semanticsofttfidf':semanticsofttfidf_score, 'sbert':sbert_score, 'bpemb':bpemb_score, 'bert':bert_score, 'match':pair['match']}, ignore_index=True)

        df_with_similarity_scores = df_with_similarity_scores.append({'osm_name':pair['osm_name'], 'yelp_name':pair['yelp_name'], 'levenshtein': levenshtein_score, 'jaro': jaro_score, 'jaro_winkler':jaro_winkler_score, 'cosine':cosine_score, 'jaccard':jaccard_score, 'tfidf':tfidf_score, 'softtfidf':softtfidf_score, 'semanticsofttfidf':semanticsofttfidf_score, 'sbert':sbert_score, 'bpemb':bpemb_score, 'bert':bert_score, 'match':pair['match']}, ignore_index=True)

    # print(df_with_similarity_scores)
    return df_with_similarity_scores

def randomForest(df):
    X = df.drop(['match'], axis=1)
    y = df['match']
    
    k=5
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    model = RandomForestClassifier(n_estimators=600 , random_state=0 , n_jobs=2, max_depth=100)
    precision = []
    recall = []
    f1 = []
    mcc = []
    
    for train_index , test_index in kf.split(X):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y[train_index] , y[test_index]
    
        X_train_without_names = X_train.drop(['osm_name', 'yelp_name'], axis=1)
        X_test_without_names = X_test.drop(['osm_name', 'yelp_name'], axis=1)
        
        model = model.fit(X_train_without_names.astype(float), y_train.astype(float))
        predictions = model.predict(X_test_without_names.astype(float))
        
        df_incorrect = pd.DataFrame(X_test['osm_name'], columns=['osm_name']) #create dataframe where similarity score can be added to pairs
        df_incorrect['yelp_name'] = X_test['yelp_name']
        df_incorrect['prediction'] = predictions
        df_incorrect['correct_label'] = y_test
        
        df_incorrect = df_incorrect[df_incorrect['prediction'] != df_incorrect['correct_label']]
        
        
        print("EVALUATION for split: ")
        
        print("=========================False positives:========================================")
        for index, pair in df_incorrect.iterrows():
            if (pair['prediction'] == 1) and (pair['correct_label'] == 0):
                print(pair['osm_name'], "    ", pair['yelp_name'], "    prediction: ", pair['prediction'], "  correct: ", pair['correct_label'])

        print("==========================Flase negatives:========================================")
        for index, pair in df_incorrect.iterrows():
            if (pair['prediction'] == 0) and (pair['correct_label'] == 1):
                print(pair['osm_name'], "    ", pair['yelp_name'], "    prediction: ", pair['prediction'], "  correct: ", pair['correct_label'])

        
        tn, fp, fn, tp = confusion_matrix(list(y_test), list(predictions), labels=[0, 1]).ravel()
        print("tn: ", tn)
        print("tp: ", tp)
        print("fp: ", fp)
        print("fn: ", fn)
        
    
        print("precision: ", precision_score(predictions, y_test.astype(float)))
        precision.append(precision_score(predictions, y_test.astype(float)))
        print("recall: ", recall_score(predictions, y_test.astype(float)))
        recall.append(recall_score(predictions, y_test.astype(float)))
        print("f1: ", f1_score(predictions, y_test.astype(float)))
        f1.append(f1_score(predictions, y_test.astype(float)))
        print("mcc: ", matthews_corrcoef(predictions, y_test.astype(float)))
        mcc.append(matthews_corrcoef(predictions, y_test.astype(float)))
     
    avg_precision = sum(precision)/k
    avg_recall = sum(precision)/k
    avg_f1 = sum(f1)/k
    avg_mcc = sum(mcc)/k
    
    return avg_precision, avg_recall, avg_f1, avg_mcc
    #return precision_score(predictions, y_test.astype(float)), recall_score(predictions, y_test.astype(float)), f1_score(predictions, y_test.astype(float)), matthews_corrcoef(predictions, y_test.astype(float))


def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    df = load_df()
    df_with_similarity_metrics = createDataFrame(df)
    precision, recall, f1, mcc = randomForest(df_with_similarity_metrics)
    print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)
    
    

if __name__ == "__main__":
    main()
