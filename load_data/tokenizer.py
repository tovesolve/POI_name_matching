import datetime
import pandas as pd
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.tokenize.api import StringTokenizer, TokenizerI
import re


pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

def tokenize_on_space(string):
    token_list = word_tokenize(string)
    tokens = {t for t in token_list} 
    return tokens
        
def tokenize_name(name):
    return word_tokenize(name)

def tokenize_dataframe(df):
    data_colnames = ['osm_token', 'yelp_token']
    df_tokens = pd.DataFrame(columns=data_colnames)
    
    for index, pair in df.iterrows():
        osm_wo_spec = remove_special_characters(pair['osm_name'])
        yelp_wo_spec = remove_special_characters(pair['yelp_name'])

        osm_tokenized = tokenize_name(osm_wo_spec)
        yelp_tokenized = tokenize_name(yelp_wo_spec)

        osm_tokenized = tokens_to_lower(osm_tokenized)
        yelp_tokenized = tokens_to_lower(yelp_tokenized)

        df_tokens = df_tokens.append({'osm_token': osm_tokenized, 'yelp_token': yelp_tokenized}, ignore_index=True)

    df['osm_token'] = df_tokens['osm_token']
    df['yelp_token'] = df_tokens['yelp_token']  

    return df

#tokens to lower case
def tokens_to_lower(token_list):
    return [token.lower() for token in token_list]

#define and remove special characters
def remove_special_characters(name):
    pattern = r'[!,*)@#%(&$_?.^\'’´]'
    mod_name = re.sub(pattern, '', name)

    pattern = r'[é]'
    mod_name = re.sub(pattern, 'e', mod_name)
    return mod_name


def main():
    df = pd.read_pickle('df_pairs_1645521936.8028078.pkl')
    df = tokenize_dataframe(df)
    df.to_pickle('./df_w_tokens' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + '.pkl') # save dataframe
    print(df)

if __name__ == "__main__":
    main()