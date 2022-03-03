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
    #f = sns.heatmap(cm, annot=True)
    #print(f)
    #examplePlot()
    #plot_confusion_matrix(df['match'].tolist(), df['score'].tolist(), classes=[0,1], title="cm")
    example()
    display_labels=[0,1]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot()
    plt.show()
    return precision, recall, f1, matthew

def get_confusion_matrix(df):
    return confusion_matrix(df['match'].tolist(), df['score'].tolist())
    
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

def example():
    # x axis values 
    x = [1,2,3] 
    # corresponding y axis values 
    y = [2,4,1] 
        
    # plotting the points  
    plt.plot(x, y) 
        
    # naming the x axis 
    plt.xlabel('x - axis') 
    # naming the y axis 
    plt.ylabel('y - axis') 
        
    # giving a title to my graph 
    plt.title('My first graph!') 
        
    # function to show the plot 
    plt.show()     
    
def main():
    pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

    #df = pd.read_pickle('v0_df_pairs_florida2022-02-28.094015.pkl')
    df = pd.read_pickle('df_pairs_boston2022-02-28.110406.pkl')

if __name__ == "__main__":
    main()
