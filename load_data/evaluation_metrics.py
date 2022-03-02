from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from drop_label import drop_rows_with_label
import matplotlib.pyplot as plt
 
def get_metrics(df, threshold):
    for index, row in df.iterrows():
        if float(row['score']) >= float(threshold):
            #print("above threshold: ", row['osm_name'], row['yelp_name'], row['match'], row['score'])
            df.at[index, 'score'] = 1
        else:
            df.at[index, 'score'] = 0
    
    #print(df)
    #precision: 
    precision = precision_score(df['match'].tolist(), df['score'].tolist())
    recall = recall_score(df['match'].tolist(), df['score'].tolist())
    f1 = f1_score(df['match'].tolist(), df['score'].tolist())
    return precision, recall, f1
    
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
    df = pd.read_pickle('df_pairs_boston2022-02-28.110406.pkl')

if __name__ == "__main__":
    main()
