from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
import numpy as np
import pandas as pd
from drop_label import drop_rows_with_label
import matplotlib.pyplot as plt
import seaborn as sns
 
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
    cm = get_confusion_matrix(df)
    display_confusion_matrix(cm)
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
    plt.show()
    
    # saves the plot to a png file
    fig, _ = plt.subplots()
    fig.savefig("eval.png")

def plot_evaluation_graph(metrics_object_dict, threshold_list, sim_func_list, metric):
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

    X_axis = np.arange(len(threshold_list))
    nbr_of_funcs = len(sim_func_list)
    width = 0.2
    total_width = width * nbr_of_funcs
    start = X_axis - total_width/2 + width/2
    for sim_func in sim_func_list:
        f1_value = metrics_object_dict[sim_func]
        plt.bar(start, f1_value, width, label = str(sim_func.__name__))
        for i in range(len(f1_value)):
            plt.text(start[i], f1_value[i] + 0.01, str(round(f1_value[i],2)))
        start = start + width

    plt.xticks(X_axis, threshold_list)
    plt.xlabel("threshold values")
    plt.ylabel(str(metric))
    plt.title(str(metric))
    plt.grid(axis='y')
    plt.legend()
    plt.show()
    
    # saves the plot to a png file
    fig, _ = plt.subplots()
    fig.savefig("eval.png")
   
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    #df = pd.read_pickle('df_pairs_boston2022-02-28.110406.pkl')
    #f1_comparision_graph(metrics_object_list={'levenstein':[0.31,20,30], 'damarau': [10, 10, 10], 'jaro': [5,5,5]}, threshold_list= [1,2,3], sim_func_list=['levenstein', 'damarau', 'jaro'])
    #f1_comparision_graph(metrics_object_list={'levenstein':[10,20,30], 'damarau': [10, 10, 10]}, threshold_list= [1,2,3], sim_func_list=['levenstein', 'damarau'])

if __name__ == "__main__":
    main()



