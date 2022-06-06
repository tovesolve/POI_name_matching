from re import X
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from word_embeddings_cosine import *
from sklearn.model_selection import train_test_split
import character_based_func
import tensorflow as tf
from tensorflow import keras
import token_based_func
import test_hybrid_func
import semantic_soft_tfidf
import restricted_softtfidf
from bpemb import BPEmb as BPEmbedding
import word_embeddings_cosine
import word_embeddings
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from tokenizer import tokenize
import xgboost
import shap
import pickle

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

        #ändra modellen och get_embedding i calc-metoden
        semanticsofttfidf_score = semantic_soft_tfidf.calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.85, character_based_func.jaro_winkler_similarity, 0.7, bpemb_model, document_frequency, tokenizer_BERT) #dessa thresholds är med det bästa vi testat hittils, ev uppdatera. (tokenizer BERT används inte)
        sbert_score = sklearn_cosine_similarity(word_embeddings.get_embedding_sBERT(pair['osm_name'], sBERT_model), word_embeddings.get_embedding_sBERT(pair['yelp_name'], sBERT_model))[0][0]
        bpemb_score = sklearn_cosine_similarity(word_embeddings.get_embedding_BPEmb(pair['osm_name'], bpemb_model), word_embeddings.get_embedding_BPEmb(pair['yelp_name'], bpemb_model))[0][0]
        bert_score = sklearn_cosine_similarity(word_embeddings.get_embedding_BERT(pair['osm_name'], BERT_model, tokenizer_BERT), word_embeddings.get_embedding_BERT(pair['yelp_name'], BERT_model, tokenizer_BERT))[0][0]
        # df_with_similarity_scores = df_with_similarity_scores.append({'levenshtein': levenshtein_score, 'jaro': jaro_score, 'jaro_winkler':jaro_winkler_score, 'cosine':cosine_score, 'jaccard':jaccard_score, 'tfidf':tfidf_score, 'match':pair['match']}, ignore_index=True)
        # df_with_similarity_scores = df_with_similarity_scores.append({'osm_name':pair['osm_name'], 'yelp_name':pair['yelp_name'], 'softtfidf':softtfidf_score, 'semanticsofttfidf':semanticsofttfidf_score, 'sbert':sbert_score, 'bpemb':bpemb_score, 'bert':bert_score, 'match':pair['match']}, ignore_index=True)

        df_with_similarity_scores = df_with_similarity_scores.append({'osm_name':pair['osm_name'], 'yelp_name':pair['yelp_name'], 'levenshtein': levenshtein_score, 'jaro': jaro_score, 'jaro_winkler':jaro_winkler_score, 'cosine':cosine_score, 'jaccard':jaccard_score, 'tfidf':tfidf_score, 'softtfidf':softtfidf_score, 'semanticsofttfidf':semanticsofttfidf_score, 'sbert':sbert_score, 'bpemb':bpemb_score, 'bert':bert_score, 'match':pair['match']}, ignore_index=True)

    # print(df_with_similarity_scores)
    return df_with_similarity_scores

def validateModel(X_train, y_train, model, seed):
    k=5
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    precision = []
    recall = []
    f1 = []
    mcc = []
    tn_tot=0
    fp_tot=0
    fn_tot=0
    tp_tot=0

    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_fp_fn = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs

    for train_index , test_index in kf.split(X_train):
        X_train_fold , X_val_fold = X_train.iloc[train_index,:],X_train.iloc[test_index,:]
        y_train_fold , y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        X_train_fold_without_names = X_train_fold.drop(['osm_name', 'yelp_name'], axis=1)
        X_val_fold_without_names = X_val_fold.drop(['osm_name', 'yelp_name'], axis=1)

        model = model.fit(X_train_fold_without_names.astype(float), y_train_fold.astype(float))
        predictions = model.predict(X_val_fold_without_names.astype(float))

        df_incorrect = pd.DataFrame(X_val_fold['osm_name'], columns=['osm_name']) #create dataframe where similarity score can be added to pairs
        df_incorrect['yelp_name'] = X_val_fold['yelp_name']
        df_incorrect['prediction'] = predictions
        df_incorrect['correct_label'] = y_val_fold

        df_incorrect = df_incorrect[df_incorrect['prediction'] != df_incorrect['correct_label']]
        df_fp_fn = pd.concat([df_fp_fn, df_incorrect])
        
        print("EVALUATION for split: ")
        
        print("=========================False positives:========================================")
        for index, pair in df_incorrect.iterrows():
            if (pair['prediction'] == 1) and (pair['correct_label'] == 0):
                print(pair['osm_name'], "    ", pair['yelp_name'], "    prediction: ", pair['prediction'], "  correct: ", pair['correct_label'])


        print("==========================Flase negatives:========================================")
        for index, pair in df_incorrect.iterrows():
            if (pair['prediction'] == 0) and (pair['correct_label'] == 1):
                print(pair['osm_name'], "    ", pair['yelp_name'], "    prediction: ", pair['prediction'], "  correct: ", pair['correct_label'])
        
        tn, fp, fn, tp = confusion_matrix(list(y_val_fold), list(predictions), labels=[0, 1]).ravel()
        tn_tot = tn_tot+tn
        fp_tot = fp_tot+fp
        fn_tot = fn_tot+fn
        tp_tot = tp_tot+tp
        # print("tn: ", tn)
        # print("tp: ", tp)
        # print("fp: ", fp)
        # print("fn: ", fn)
    
        #print("precision: ", precision_score(predictions, y_val_fold.astype(float)))
        precision.append(precision_score(predictions, y_val_fold.astype(float)))
        #print("recall: ", recall_score(predictions, y_val_fold.astype(float)))
        recall.append(recall_score(predictions, y_val_fold.astype(float)))
        #print("f1: ", f1_score(predictions, y_val_fold.astype(float)))
        f1.append(f1_score(predictions, y_val_fold.astype(float)))
        #print("mcc: ", matthews_corrcoef(predictions, y_val_fold.astype(float)))
        mcc.append(matthews_corrcoef(predictions, y_val_fold.astype(float)))


    print("averages cm:")
    print("tn: ", tn_tot)
    print("fp: ", fp_tot)
    print("fn: ", fn_tot)
    print("tp: ", tp_tot)

    avg_precision = sum(precision)/k
    avg_recall = sum(precision)/k
    avg_f1 = sum(f1)/k
    avg_mcc = sum(mcc)/k
    
    print()

    return avg_precision, avg_recall, avg_f1, avg_mcc


def testModel(X_train, y_train, X_test, y_test, model):
    X_train_without_names = X_train.drop(['osm_name', 'yelp_name'], axis=1)
    X_test_without_names = X_test.drop(['osm_name', 'yelp_name'], axis=1)

    model = model.fit(X_train_without_names.astype(float), y_train.astype(float))
    predictions = model.predict(X_test_without_names.astype(float))

    return precision_score(predictions, y_test.astype(float)), recall_score(predictions, y_test.astype(float)), f1_score(predictions, y_test.astype(float)), matthews_corrcoef(predictions, y_test.astype(float)), model

def predictLoadedModel(X_train, y_train, X_test, y_test, model):
    #X_train_without_names = X_train.drop(['osm_name', 'yelp_name'], axis=1)
    X_test_without_names = X_test.drop(['osm_name', 'yelp_name'], axis=1)
    predictions = model.predict(X_test_without_names.astype(float))
    return predictions, precision_score(predictions, y_test.astype(float)), recall_score(predictions, y_test.astype(float)), f1_score(predictions, y_test.astype(float)), matthews_corrcoef(predictions, y_test.astype(float)), model
   
def evaluateModel(X_test, y_test, predictions):
    #data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    #df_fp_fn = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs    
    df_incorrect = pd.DataFrame(X_test['osm_name'], columns=['osm_name']) #create dataframe where similarity score can be added to pairs
    df_incorrect['yelp_name'] = X_test['yelp_name']
    df_incorrect['prediction'] = predictions
    df_incorrect['correct_label'] = y_test

    df_incorrect = df_incorrect[df_incorrect['prediction'] != df_incorrect['correct_label']]
    #df_fp_fn = pd.concat([df_fp_fn, df_incorrect])
    
    print("EVALUATION for model: ")
    
    print("=========================False positives: (non-matches)========================================")
    for index, pair in df_incorrect.iterrows():
        if (pair['prediction'] == 1) and (pair['correct_label'] == 0):
            print(pair['osm_name'], "    ", pair['yelp_name'], "    prediction: ", pair['prediction'], "  correct: ", pair['correct_label'])

    print("==========================False negatives: (matches)========================================")
    for index, pair in df_incorrect.iterrows():
        if (pair['prediction'] == 0) and (pair['correct_label'] == 1):
            print(pair['osm_name'], "    ", pair['yelp_name'], "    prediction: ", pair['prediction'], "  correct: ", pair['correct_label'])
    
    tn, fp, fn, tp = confusion_matrix(list(y_test), list(predictions), labels=[0, 1]).ravel()
    
    print("confusion matrix: ")
    print("tn: ", tn)
    print("tp: ", tp)
    print("fp: ", fp)
    print("fn: ", fn)
    
    #plotRandomForest(model, X_train, X_test)
    #plotGradientBoost(model, X_train, X_test)
    #plotNeuralNetwork(model, X_train, X_test)
    


def plotRandomForest(model, X_train, X_test):
    X_train_without_names = X_train.drop(['osm_name', 'yelp_name'], axis=1).astype(float)
    X_test_without_names = X_test.drop(['osm_name', 'yelp_name'], axis=1).astype(float)

    #samples = shap.sample(X_train, 10)
    #explainer = shap.KernelExplainer(model.predict, samples)
    #shap_values = explainer.shap_values(X_test) #ta ut shap-values för varje split

    explainer = shap.TreeExplainer(model, X_train_without_names, model_output='probability')
    shap_values = explainer.shap_values(X_test_without_names, check_additivity=False) #ta ut shap-values för varje split

    # shap.summary_plot(shap_values[0], X_test_without_names)

    # fig = plt.gcf()
    # plt.show()
    # plt.draw()

    # #byt namn på den här
    # img_name = 'ml_plots/dot_plot_random_forest' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    # fig.set_size_inches(10, 6)
    # fig.savefig(img_name, dpi=100)
    # plt.clf()

    #bar plot using TreeExplainer
    #explainer = shap.TreeExplainer(model, X_train_without_names)
    #shap_values = explainer.shap_values(X_test_without_names, check_additivity=False) #ta ut shap-values för varje split

    #print("random forest shap: ", shap_values[0])

    #bar plot using KernelExplainer
    #shap.summary_plot(shap_values, X_test, plot_type="bar")
    shap.summary_plot(shap_values[0], X_test_without_names, plot_type="bar")

    fig = plt.gcf()
    plt.show()
    plt.draw()

    #byt namn på den här
    img_name = 'ml_plots/bar_plot_random_forest' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.set_size_inches(16, 6)
    fig.savefig(img_name, dpi=100)

    plt.clf()

def plotGradientBoost(model, X_train, X_test):
    X_train_without_names = X_train.drop(['osm_name', 'yelp_name'], axis=1).astype(float)
    X_test_without_names = X_test.drop(['osm_name', 'yelp_name'], axis=1).astype(float)


    explainer = shap.TreeExplainer(model, X_train_without_names, model_output='probability')
    shap_values = explainer.shap_values(X_test_without_names, check_additivity=False) #ta ut shap-values för varje split

    #print("gradient boost shap: ", shap_values)

    # Dot plot
    # shap.summary_plot(shap_values, X_test_without_names)

    # fig = plt.gcf()
    # plt.show()
    # plt.draw()

    # #byt namn på den här
    # img_name = 'ml_plots/dot_plot_gradient_boost_bpemb_default_whole_dataset' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    # fig.savefig(img_name, dpi=100)

    # plt.clf()

    # bar plot
    shap.summary_plot(shap_values, X_test_without_names, plot_type="bar")

    fig = plt.gcf()
    plt.show()
    plt.draw()

    #byt namn på den här
    img_name = 'ml_plots/bar_plot_gradient_boost' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.set_size_inches(16, 6)
    fig.savefig(img_name, dpi=100)
    plt.clf()


def plotNeuralNetwork(model, predictions, X_train, X_test):
    X_train_without_names = X_train.drop(['osm_name', 'yelp_name'], axis=1).astype(float)
    X_test_without_names = X_test.drop(['osm_name', 'yelp_name'], axis=1).astype(float)
    
    explainer = shap.KernelExplainer(model.predict, X_train_without_names)
    shap_values = explainer.shap_values(X_test_without_names) #ta ut shap-values för varje split

    # Dot plot
    #shap.summary_plot(shap_values, X_test_without_names)

    # fig = plt.gcf()
    # plt.show()
    # plt.draw()

    # #byt namn på den här
    # img_name = 'dot_plot_neural_network_bpemb_default_whole_dataset' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    # fig.savefig(img_name, figsize=(8, 6), dpi=100)

    plt.clf()

    # bar plot
    shap.summary_plot(shap_values, X_test_without_names, plot_type="bar")

    fig = plt.gcf()
    plt.show()
    plt.draw()

    #byt namn på den här
    img_name = 'ml_plots/bar_plot_neural_network' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.set_size_inches(16, 6)
    fig.savefig(img_name, dpi=100)
    plt.clf()


def keras_neuralnetwork(df):
    X = df.drop(['match'], axis=1)
    y = df['match']

    k=5
    kf = KFold(n_splits=k, random_state=None, shuffle=True)
    #model = RandomForestClassifier(n_estimators=600 , random_state=0 , n_jobs=2, max_depth=100) #santos params
    model = keras.Sequential([
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=192, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    precision = []
    recall = []
    f1 = []
    mcc = []

    data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match', 'score']
    df_fp_fn = pd.DataFrame(columns=data_colnames) #create dataframe where similarity score can be added to pairs

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train = X_train.drop(['osm_name', 'yelp_name'], axis=1)
    X_test = X_test.drop(['osm_name', 'yelp_name'], axis=1)



    model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model.fit(X_train, y_train)

    model.evaluate(X_test,  y_test, verbose=2)


def temp_update_df(df):

    corpus_list, document_frequency = prepareCorpus(df)
    bpemb_model = BPEmbedding(lang="en", dim=300, vs=50000)
    sBERT_model = SentenceTransformer("all-mpnet-base-v2")
    tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    BERT_model = BertModel.from_pretrained("bert-base-uncased")

    semanticsofttfidf_score_BERT_list = []
    semanticsofttfidf_score_BPEmb_list = []

    for index, pair in df.iterrows():
        #ändra modellen och get_embedding i calc-metoden
        semanticsofttfidf_score_BPEmb = semantic_soft_tfidf.calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.85, character_based_func.jaro_winkler_similarity, 0.7, bpemb_model, document_frequency, tokenizer_BERT) #dessa thresholds är med det bästa vi testat hittils, ev uppdatera. (tokenizer BERT används inte)
        semanticsofttfidf_score_BPEmb_list.append(semanticsofttfidf_score_BPEmb)
        #semanticsofttfidf_score_BERT = semantic_soft_tfidf.calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.85, character_based_func.jaro_winkler_similarity, 0.95, BERT_model, document_frequency, tokenizer_BERT) #dessa thresholds är med det bästa vi testat hittils, ev uppdatera. (tokenizer BERT används inte)
        #semanticsofttfidf_score_BERT_list.append(semanticsofttfidf_score_BERT)

    #df['semanticsofttfidf_BERT'] = semanticsofttfidf_score_BERT_list
    df['semanticsofttfidf_BPEmb'] = semanticsofttfidf_score_BPEmb_list
    print(df)

        # df_with_similarity_metrics = temp_update_df(df_with_similarity_metrics)
    # df_with_similarity_metrics.to_pickle('./similarity_mertics_df_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    # print('saved')
    # with pd.ExcelWriter("df_sim_metrics_pickle.xlsx") as writer:
    #     df_with_similarity_metrics.to_excel(writer)


    return df

def update_softTFIDF():
    df = load_df() #alla POIs 
    corpus_list, document_frequency = prepareCorpus(df)

    df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_2022-05-13.112436.pkl') # load saved df with features

    soft_scores = []
    for index, pair in df.iterrows():
        softtfidf_score = test_hybrid_func.calc_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.85, character_based_func.jaro_winkler_similarity, document_frequency)
        soft_scores.append(softtfidf_score)
    
    print("shape before: ", df_with_similarity_metrics.shape)
    df_with_similarity_metrics['softtfidf'] = soft_scores
    print("shape after: ", df_with_similarity_metrics.shape)
    df_with_similarity_metrics.to_pickle('./similarity_mertics_df_w_distance_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    print('saved')
    
    with pd.ExcelWriter("df_sim_metrics_pickle_2022_06_03.xlsx") as writer:
        df_with_similarity_metrics.to_excel(writer)


def add_tokencount_to_df(df):
    token_counts_osm = []
    token_counts_yelp = []
    token_counts_ratio = []
    for index, pair in df.iterrows():
        len_osm_name = len(tokenize(pair['osm_name']))
        len_yelp_name = len(tokenize(pair['yelp_name']))
        ratio = max(len_osm_name, len_yelp_name)/min(len_osm_name, len_yelp_name)
        # print(pair['osm_name'], " length: ", len_osm_name)
        # print(pair['yelp_name'], " length: ", len_yelp_name)
        # print("ratio: ", ratio)
        token_counts_osm.append(len_osm_name)
        token_counts_yelp.append(len_yelp_name)
        token_counts_ratio.append(ratio)
    
    #df['tlen_osm'] = token_counts_osm
    #df['tlen_yelp'] = token_counts_yelp
    df['tlen_ratio'] = token_counts_ratio
    print(df)

    df.to_pickle('./similarity_mertics_df_w_tokencount' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    print('saved')
    # with pd.ExcelWriter("df_sim_metrics_pickle.xlsx") as writer:
    #     df_with_similarity_metrics.to_excel(writer)

def add_we_to_df(df):
    bpemb_model = BPEmbedding(lang="en", dim=50, vs=50000)
    #sBERT_model = SentenceTransformer("all-mpnet-base-v2")
    #tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    #BERT_model = BertModel.from_pretrained("bert-base-uncased")

    BPEmb_list = []

    for index, pair in df.iterrows():
        emb1_BPEmb = get_embedding_BPEmb(pair['osm_name'], bpemb_model)
        emb2_BPEmb = get_embedding_BPEmb(pair['yelp_name'], bpemb_model)
        #print(pair['osm_name'], " ", pair['yelp_name'])
        #rint(emb1_BPEmb)
        #print(emb2_BPEmb)
        v = np.subtract(emb2_BPEmb, emb1_BPEmb)
        #print("v: ", v)
        for i in range(0, len(v[0])):
            #print('v[i] ', v[0][i])
            #print(i)
            df.at[index, 'dim' + str(i)] = v[0][i]
        #print(df.at[index])
    #print(df)

        # df_with_similarity_metrics = temp_update_df(df_with_similarity_metrics)
    # df_with_similarity_metrics.to_pickle('./similarity_mertics_df_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    # print('saved')
    # with pd.ExcelWriter("df_sim_metrics_pickle.xlsx") as writer:
    #     df_with_similarity_metrics.to_excel(writer)


    return df

def add_distance_df():
    df = load_df()
    df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_w_tokencount2022-05-05.133508.pkl') # load saved df with features
    
    distances = []
    for index, pair in df.iterrows():
        distances.append(pair['distance'])
    
    df_with_similarity_metrics['distance'] = distances
    print(df_with_similarity_metrics)
    df_with_similarity_metrics.to_pickle('./similarity_mertics_df_w_distance_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    print('saved')

def add_restricted_scores_to_pickle():
    df = load_df()
    df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_w_distance_2022-05-05.152614.pkl') # load saved df with features
    print("Shape for df_with_similarity_metrics: ", df_with_similarity_metrics.shape)
    
    restrictedsofttfidf_score_list = []
    for index, pair in df.iterrows():
        data_colnames = ['osm_name', 'yelp_name', 'osm_latitude', 'osm_longitude', 'yelp_latitude', 'yelp_longitude', 'distance', 'match']
        df_restricted = pd.DataFrame(columns=data_colnames) #create dataframe with nearby POIs
        original_osm, original_yelp = pair['osm_name'], pair['yelp_name']
        for i, p in df.iterrows():
            if p['osm_name'] == original_osm or p['yelp_name'] == original_yelp:
                df_restricted = df_restricted.append({'osm_name': p['osm_name'], 'yelp_name': p['yelp_name'], 'osm_latitude': p['osm_latitude'], 'osm_longitude': p['osm_longitude'], 'yelp_latitude': p['yelp_latitude'], 'yelp_longitude': p['yelp_longitude'], 'distance': p['distance'], 'match': p['match']}, ignore_index=True)
        
        corpus_list, document_frequency = prepareCorpus(df_restricted) #create corpus from dataframe with nearby POIs
        #print("corpus list for ", pair['osm_name'], " and ", pair['yelp_name'], ": ", corpus_list)

        score = restricted_softtfidf.calc_restricted_softTFIDF_for_pair(pair['osm_name'], pair['yelp_name'], corpus_list, 0.95, character_based_func.jaro_winkler_similarity, document_frequency)
        restrictedsofttfidf_score_list.append(score)

    df_with_similarity_metrics['restricted_softtfidf'] = restrictedsofttfidf_score_list
    print("Shape for df after adding new column: ", df_with_similarity_metrics.shape)
    df_with_similarity_metrics.to_pickle('./similarity_mertics_df_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    print('saved')
    with pd.ExcelWriter("df_sim_metrics_pickle.xlsx") as writer:
        df_with_similarity_metrics.to_excel(writer)

    return df



def old_plotGradientBoost(model, X_train, X_test):
    explainer = shap.TreeExplainer(model, X_train) 
    shap_values = explainer.shap_values(X_test) #ta ut shap-values för varje split
    
    # Dot plot
    shap.summary_plot(shap_values, X_test)
    
    fig = plt.gcf()
    plt.show()
    plt.draw()
    
    #byt namn på den här
    img_name = 'dot_plot_gradient_boost_bpemb_default_whole_dataset' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.savefig(img_name, dpi=100)
    
    plt.clf()

    # bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    fig = plt.gcf()
    plt.show()
    plt.draw()
    
    #byt namn på den här
    img_name = 'bar_plot_gradient_bpemb_default_whole_dataset' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.savefig(img_name, dpi=100)
    plt.clf()
    

def main_for_update():
    update_softTFIDF()
    print("Done")

def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe
    #df = load_df()
    #df_with_similarity_metrics = createDataFrame(df)

    #create and save feature matrix with similarity scores:
    # df_with_similarity_metrics = createDataFrame(df)
    # df_with_similarity_metrics.to_pickle('./similarity_mertics_df_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.pkl') # save dataframe to pickle
    # print('saved')
    #load feature matrix with similarity scores:
    #df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_w_distance_2022-05-05.152614.pkl') # load saved df with features
        
        
    #df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_2022-05-13.112436.pkl')
    df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_w_distance_2022-06-03.165524.pkl') # load saved df with features
    
    # UNcommented rows are dropped (not included in feature vector)
    # Commented rows are included in feature vector.
    
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bert'], axis=1)
    # # df_with_similarity_metrics = df_with_similarity_metrics.drop(['cosine'], axis=1)
    # # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro_winkler'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaccard'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['levenshtein'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf'], axis=1)
    # #df_with_similarity_metrics = df_with_similarity_metrics.drop(['softtfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['sbert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tfidf'], axis=1)
    # #df_with_similarity_metrics = df_with_similarity_metrics.drop(['bpemb'], axis=1)
    
    df_with_similarity_metrics = df_with_similarity_metrics.drop(['tlen_osm'], axis=1)
    df_with_similarity_metrics = df_with_similarity_metrics.drop(['tlen_yelp'], axis=1)
    df_with_similarity_metrics = df_with_similarity_metrics.drop(['tlen_ratio'], axis=1)
    df_with_similarity_metrics = df_with_similarity_metrics.drop(['distance'], axis=1)
    
    #v1:
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaccard'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['levenshtein'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['softtfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['sbert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bpemb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['restricted_softtfidf'], axis=1)
    
    #v2
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaccard'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['levenshtein'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['softtfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['sbert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['restricted_softtfidf'], axis=1)
    
    #v3
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaccard'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['levenshtein'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['sbert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['restricted_softtfidf'], axis=1)

    #v4
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bert'], axis=1)
    
    #v5 
    #df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    #df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    
    #v6: 
    #ingen droppad
    
    #v7 
    #df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    #df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    
    X = df_with_similarity_metrics.drop(['match'], axis=1)
    y = df_with_similarity_metrics['match']
    
    seed = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)
    rf_model = RandomForestClassifier(n_estimators=300, criterion="entropy", random_state=seed) # ha med seed på modellen också
    xgb_model = xgboost.XGBClassifier(random_state=seed)
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50, 30, 20), batch_size=400, random_state=seed) #best achieved results for validation data seed=0
    #model = MLPClassifier(random_state=seed) #default
    
    
    # presentera hur vi kommit fram till hyper parametrarna eller bara förklara hur vi gör? räcker med att förklara hur vi gjort.
    
    # training and validation
    # precision, recall, f1, mcc = validateModel(X, y, rf_model, seed)
    # print("RF")
    # print("evaluation metrics for validation data (average for folds):")
    # print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)
    # precision, recall, f1, mcc = validateModel(X, y, xgb_model, seed)
    # print("XGB")
    # print("evaluation metrics for validation data (average for folds):")
    # print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)
    # precision, recall, f1, mcc = validateModel(X, y, mlp_model, seed)
    # print("MLP")
    # print("evaluation metrics for validation data (average for folds):")
    # print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)
    
    # train, predict and save model:
    print("RF")
    precision, recall, f1, mcc, model = testModel(X_train, y_train, X_test, y_test, rf_model)
    print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)  
    precision, recall, f1, mcc, model = testModel(X_train, y_train, X_test, y_test, xgb_model)
    print("XGB")
    print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)  
    precision, recall, f1, mcc, model = testModel(X_train, y_train, X_test, y_test, mlp_model)
    print("MLP")
    print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)  

    #pickle.dump(model, open('ML_similarity_metrics_RF_8.pkl', 'wb'))
    
    
    
    # load trained model
    
    #model = pickle.load(open('ML_similarity_metrics_XGB_5.pkl', 'rb'))
    
    #print("model: ", model)
    
    #predict loaded model, måste avkommentera de droppade raderna för vX ovan för att få rätt shape
    #predictions, precision, recall, f1, mcc, model = predictLoadedModel(X_train, y_train, X_test, y_test, model)
    #print("evaluation metrics for test data:")
    #print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)    
    #evaluateModel(X_test, y_test, predictions)
    
    #plotRandomForest(rf_model, X_train, X_test)
    #plotGradientBoost(xgb_model, X_train, X_test)
    #plotNeuralNetwork(model, predictions, X_train, X_test)
    
def temp_main():
    df_with_similarity_metrics = pd.read_pickle('similarity_mertics_df_w_distance_2022-05-05.152614.pkl') # load saved df with features

    # UNcommented rows are dropped (not included in feature vector)
    # Commented rows are included in feature vector.
    
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BPEmb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf_BERT'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['cosine'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro_winkler'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaccard'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['jaro'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['levenshtein'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['semanticsofttfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['softtfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['sbert'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tfidf'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['bpemb'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tlen_osm'], axis=1)
    # df_with_similarity_metrics = df_with_similarity_metrics.drop(['tlen_yelp'], axis=1)
    df_with_similarity_metrics = df_with_similarity_metrics.drop(['distance'], axis=1)

    df_with_similarity_metrics = add_we_to_df(df_with_similarity_metrics)
    
    X = df_with_similarity_metrics.drop(['match'], axis=1)
    y = df_with_similarity_metrics['match']
    
    seed = 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    model = RandomForestClassifier(n_estimators=300, criterion="entropy", random_state=seed)
    #model = xgboost.XGBClassifier(random_state=seed)
    #model = MLPClassifier(hidden_layer_sizes=(100, 50, 30, 20), batch_size=400, random_state=seed) #best achieved results for validation data seed=0
    #model = MLPClassifier(random_state=seed) #default
    print("model: ", model)
    
    precision, recall, f1, mcc = validateModel(X_train, y_train, model, seed)
    print("evaluation metrics for validation data (average for folds):")
    print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)
    
    #test:
    # precision, recall, f1, mcc, model = testModel(X_train, y_train, X_test, y_test, model)
    # print("evaluation metrics for test data:")
    # print("precision: ", precision, " recall: ", recall, " f1: ", f1, " mcc: ", mcc)
    
    #plotRandomForest(model, X_train, X_test)


if __name__ == "__main__":
    main()

