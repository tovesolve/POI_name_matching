from tkinter import CENTER
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
import numpy as np
import pandas as pd
from drop_label import drop_rows_with_label
import matplotlib.pyplot as plt
import seaborn as sns
 
def score_to_label(df, threshold):
    for index, row in df.iterrows():
        if float(row['score']) >= float(threshold):
            #print("above threshold: ", row['osm_name'], row['yelp_name'], row['match'], row['score'])
            df.at[index, 'score'] = 1
        else:
            df.at[index, 'score'] = 0
    return df

def get_metrics(df):
    precision = precision_score(df['match'].tolist(), df['score'].tolist())
    recall = recall_score(df['match'].tolist(), df['score'].tolist())
    f1 = f1_score(df['match'].tolist(), df['score'].tolist())
    matthew = matthews_corrcoef(df['match'].tolist(), df['score'].tolist())
    cm = get_confusion_matrix(df)
    #display_confusion_matrix(cm)
    return precision, recall, f1, matthew

def get_confusion_matrix(df):
    return confusion_matrix(df['match'].tolist(), df['score'].tolist())

def display_confusion_matrix(cm):
    display_labels=[0,1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.show()

def f1_comparision_graph(metrics_object_list, threshold_list, sim_func_list):
    X = threshold_list #thresholds
    
    X_axis = np.arange(len(X))

    nbr_of_funcs = len(sim_func_list)
    width = 0.2
    total_width = width * nbr_of_funcs
    start = X_axis - total_width/2 + width/2
    for sim_func in sim_func_list:
        f1_value = metrics_object_list[sim_func]
        plt.bar(start, f1_value, width, label = str(sim_func.__name__))
        for i in range(len(f1_value)):
            plt.text(start[i], f1_value[i] + 0.01, str(round(f1_value[i],2)))
        start = start + width

    plt.xticks(X_axis, X)
    plt.xlabel("Similarity functions")
    plt.ylabel("f1-score")
    plt.title("f1-score of different similarity functions")
    plt.grid(axis='y')
    plt.legend()
    plt.show()

def examplePlot():
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("test.png")
    plt.show()
   
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    #df = pd.read_pickle('df_pairs_boston2022-02-28.110406.pkl')
    f1_comparision_graph(metrics_object_list={'levenstein':[0.31,20,30], 'damarau': [10, 10, 10], 'jaro': [5,5,5]}, threshold_list= [1,2,3], sim_func_list=['levenstein', 'damarau', 'jaro'])
    f1_comparision_graph(metrics_object_list={'levenstein':[10,20,30], 'damarau': [10, 10, 10]}, threshold_list= [1,2,3], sim_func_list=['levenstein', 'damarau'])

    #examplePlot()

if __name__ == "__main__":
    main()



