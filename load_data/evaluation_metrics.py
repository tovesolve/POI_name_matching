import imp
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
from torch import embedding
from drop_label import drop_rows_with_label
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
 
def classify_scores(df, threshold):
    """
    Classifies the scores from the given dataframe based on the given threshold.

    Parameters
    ----------
    dataframe : df
        the dataframe to be classified
    threshold : float
        the threshold value to classify based on

    Returns
    -------
    dataframe
        a dataframe with classified values from the similarity scores and threshold.
    """
    
    for index, row in df.iterrows():
        if float(row['score']) >= float(threshold):
            #print("above threshold: ", row['osm_name'], row['yelp_name'], row['match'], row['score'])
            df.at[index, 'score'] = 1
        else:
            df.at[index, 'score'] = 0
    return df

def get_metrics(df):
    """
    Calculates and returns the evaluation metrics based on the classified 1/0 scores compared to the actual labels.

    Parameters
    ----------
    dataframe : df
        the dataframe to be evaluated

    Returns
    -------
    float
        precision
    float
        recall
    float
        f1_score
    float
        matthew correlation coefficient
    """
    
    precision = precision_score(df['match'].tolist(), df['score'].tolist())
    recall = recall_score(df['match'].tolist(), df['score'].tolist())
    f1 = f1_score(df['match'].tolist(), df['score'].tolist())
    matthew = matthews_corrcoef(df['match'].tolist(), df['score'].tolist())
    #cm = get_confusion_matrix(df)
    #display_confusion_matrix(cm)
    return precision, recall, f1, matthew

def get_confusion_matrix(df):
    """
    Creates the confusion matrix based on the classified 1/0 scores compared to the actual labels.

    Parameters
    ----------
    dataframe : df
        the dataframe to be evaluated, containing similarity scores

    Returns
    -------
    matrix
        the confusion matrix for the dataframe
    """
    return confusion_matrix(df['match'].tolist(), df['score'].tolist())

def display_confusion_matrix(cm):
    """
    Plots the given confusion matrix

    Parameters
    ----------
    matrix : cm
        the confusion matrix to plot
    """
    display_labels=[0,1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.title(str(""))
    fig = plt.gcf()
    plt.show()
    plt.draw()
    img_name = 'confusion_metrics_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.savefig(img_name, dpi=100)

def plot_evaluation_graph_sim_funcs(metrics_object_dict, threshold_list, sim_func_list, metric):
    """
    Creates and plots a graph for all the metrics object for given evaluation metric.

    Parameters
    ----------
    metrics_object_dict : dictionary
        the dictionary containing the similarity function with corresponsing scores.
    threshold_list : list
        the list of thresholds
    sim_func_list : list
        the list of similarity functions
    metric
        the metric to be evaluated and plotted
    """
    
    nbr_of_sim_funcs = len(sim_func_list)
    X_axis = np.arange(len(sim_func_list))

    threshold_width = 0.01
    sim_func_width = threshold_width*len(threshold_list)
    sim_func_middle_pos = (X_axis+1)/(nbr_of_sim_funcs+1)
    start_pos = sim_func_middle_pos - sim_func_width/2 + threshold_width/2
    
    ticks = sim_func_middle_pos
    colors = get_colors()
    start=start_pos
    i = 0
    func_name_list = []
    
    for s in sim_func_list:
        func_name_list.append(to_string(s.__name__))
        
    for t in threshold_list:
        scores = metrics_object_dict[t]
        plt.bar(start, scores, threshold_width, label = threshold_list[i], color=colors[i])
        for j in range(len(scores)):
            plt.text(start[j]-0.008, scores[j]+0.01, str(round(scores[j],3)), fontsize=6)
        start = start + threshold_width
        i += 1
        
    plt.xticks(ticks, func_name_list)
    plt.xlabel("Similarity Function")
    plt.ylabel(to_string(metric))
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Threshold", bbox_to_anchor=(1.08, 1.015))
    
    fig = plt.gcf()
    fig.set_size_inches(16, 6)
    plt.show()
    plt.draw()
    img_name = 'sim_funcs_' + str(metric) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.savefig(img_name, dpi=100)
    
def plot_evaluation_graph_cosine_word_embeddings(metrics_object_dict, threshold_list, word_embeddings_list, metric):
    """
    Creates and plots a graph for all the metrics object for given evaluation metric.

    Parameters
    ----------
    metrics_object_dict : dictionary
        the dictionary containing the word embedding with corresponsing scores.
    threshold_list : list
        the list of thresholds
    sim_func_list : list
        the list of word embeddings
    metric
        the metric to be evaluated and plotted
    """

    nbr_of_sim_funcs = len(word_embeddings_list)
    X_axis = np.arange(len(word_embeddings_list))

    threshold_width = 0.02
    sim_func_width = threshold_width*len(threshold_list)
    sim_func_middle_pos = (X_axis+1)/(nbr_of_sim_funcs+1)
    start_pos = sim_func_middle_pos - sim_func_width/2 + threshold_width/2
    
    ticks = sim_func_middle_pos
    colors = get_colors()
    start=start_pos
    i = 0
    func_name_list = []
    
    for s in word_embeddings_list:
        if s.__name__.lower() == "sbert":
            print("sbert")
            func_name_list.append("sBERT")
        elif s.__name__.lower() == "bert":
            func_name_list.append("BERT")
        elif s.__name__.lower() == "bpemb":
            func_name_list.append("BPEmb")
        
    for t in threshold_list:
        scores = metrics_object_dict[t]
        plt.bar(start, scores, threshold_width, label = threshold_list[i], color=colors[i])
        for j in range(len(scores)):
            plt.text(start[j]-0.008, scores[j]+0.01, str(round(scores[j],3)), fontsize=6)
        start = start + threshold_width
        i += 1

    print(func_name_list)
    plt.xticks(ticks, func_name_list)
    plt.xlabel("Word Embedding")
    plt.ylabel(to_string(metric))
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Threshold", bbox_to_anchor=(1.08, 1.015))
    
    fig = plt.gcf()
    fig.set_size_inches(15, 6)
    print(fig)
    plt.show()
    plt.draw()
    img_name = 'cosine_word_embeds_' + str(metric) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d.%H%M%S")) + '.png' #TODO save to better file name
    fig.savefig(img_name, dpi=100)
    
def to_string(metric):
    s = str(metric).capitalize()
    return re.sub(r"[_]", ' ', s)

def get_colors():
    color_list = [[0.8, 0.15, 0.15, 0.7],
                  [0.47, 0.67, 0.27, 0.7],
                  [0.93, 0.79, 0.0, 0.7],
                  [0.44, 0.58, 0.86, 0.7],
                  [0.93, 0.46, 0.13, 0.7],
                  [0.37, 0.18, 0.47, 0.7],
                  [0.21, 0.86, 0.79, 0.7],
                  [0.33, 0.10, 0.55, 0.7],
                  [0.72, 0.07, 0.14, 0.7],
                  [0.8, 0.41, 0.54, 0.7],
                  [0.23, 0.29, 0.56, 0.7]]
    return color_list
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    #df = pd.read_pickle('df_pairs_boston2022-02-28.110406.pkl')
    #f1_comparision_graph(metrics_object_list={'levenstein':[0.31,20,30], 'damarau': [10, 10, 10], 'jaro': [5,5,5]}, threshold_list= [1,2,3], sim_func_list=['levenstein', 'damarau', 'jaro'])
    #f1_comparision_graph(metrics_object_list={'levenstein':[10,20,30], 'damarau': [10, 10, 10]}, threshold_list= [1,2,3], sim_func_list=['levenstein', 'damarau'])
    #plot_evaluation_graph({'leven': [0, 0.5, 0.5]}, [0, 0.5, 1], ['leven'], 'f1_score')

if __name__ == "__main__":
    main()



