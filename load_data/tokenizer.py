import datetime
import pandas as pd
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.tokenize.api import StringTokenizer, TokenizerI
import re
from nltk.corpus import stopwords
import unidecode
from nltk.stem import SnowballStemmer


pd.set_option("display.max_rows", None, "display.max_columns", None) #show all rows when printing dataframe

def tokenize_on_space(string):
    pattern = r"\S+"
    return re.findall(pattern, string)
    
# saker att fundera på i tokenizern:
#ändra - till blank? eller bara ta bort? Hitta källa på vad som är best i engelskan?
# något sätt att hantera särskrivningar?
#borde ändra grad-tecknet också


#tokenize name on space, to lower, remove special characters
def tokenize_name(name):
    #lägger till blank mellan liten och stor bokstav, tex "proOptical" -> "pro Optical"
    #print(name)
    #tokenized = re.sub(r'(?<![A-Z\W])(?=[A-Z])', ' ', name)
    #print(tokenized)
    #tokenized = tokenized.lower()
    tokenized = name.lower() #lower case
    
    #tokenized = re.sub(r"[-._!\"`'#%&,:;<>=@{}~\$\(\)\*\+\/\\\?\[\]\^\|]+", '', tokenized) #tar bort alla dessa specialtecken
    #tokenized = re.sub(r"[éèê]", 'e', tokenized) #vill ersätta varianter på tecken till orginaltecken.
    
    tokenized = re.sub(r"[;]", ' ', tokenized)
    tokenized = unidecode.unidecode(tokenized)
    #tokenized.normalize("NFD").replace(r"/[\u0300-\u036f]/g", "")
    
    tokenized = re.sub(r"[^\sa-zA-Z0-9]", '', tokenized) #ta bort allt som inte är små, stora bokstäver eller siffror
    
    pattern = r"\S+" #splittar på mellanslag
    tokens_list = re.findall(pattern, tokenized)
    
    # find stopwords
    stop_words = stopwords.words('english')
    filtered_stopwords = filter_stopwords(stop_words)
    #print("stop words: ", filtered_stopwords)
    
    # filter token_list on stopwords
    tokens_list = [w for w in tokens_list if not w.lower() in filtered_stopwords]
    
    #stem tokens to their basic form
    stemmed_tokens_list = stem(tokens_list)
    
    return stemmed_tokens_list

def filter_stopwords(stop_words):
    filtered_stopwords = []
    for w in stop_words:
        word = re.sub(r"[']", '', w)
        if len(word) <= 3:
            filtered_stopwords.append(word)
    return filtered_stopwords

def stem(tokens_list):
    snowball = SnowballStemmer(language='english')
    stemmed_list=[]
    for token in tokens_list:
        stemmed_list.append(snowball.stem(token))
    return stemmed_list

def concat_token_list(token_list):
    word = ""
    for t in token_list:
        word+= t + " "
    return word

def tokenize(name):
    #return tokenize_on_space(name)
    return tokenize_name(name)

#används inte
# def tokenize_dataframe(df):
#     data_colnames = ['osm_token', 'yelp_token']
#     df_tokens = pd.DataFrame(columns=data_colnames)
    
#     for index, pair in df.iterrows():
#         osm_wo_spec = remove_special_characters(pair['osm_name'])
#         yelp_wo_spec = remove_special_characters(pair['yelp_name'])

#         osm_tokenized = tokenize_name(osm_wo_spec)
#         yelp_tokenized = tokenize_name(yelp_wo_spec)

#         osm_tokenized = tokens_to_lower(osm_tokenized)
#         yelp_tokenized = tokens_to_lower(yelp_tokenized)

#         df_tokens = df_tokens.append({'osm_token': osm_tokenized, 'yelp_token': yelp_tokenized}, ignore_index=True)

#     df['osm_token'] = df_tokens['osm_token']
#     df['yelp_token'] = df_tokens['yelp_token']  

#     return df

#tokens to lower case
def tokens_to_lower(token_list):
    return [token.lower() for token in token_list]

def token_to_lower(name):
    return name.lower()


def main():
    #df = pd.read_pickle('df_pairs_1645521936.8028078.pkl')
    #df = tokenize_dataframe(df)
    #df.to_pickle('./df_w_tokens' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) + '.pkl') # save dataframe
    #print(df)
    
    #stopword from nltk
    #stopwords = {'them', 'about', 'hadn', 'mustn', 'such', 'can', 'isn', 'most', 'haven', 'once', 'more', "don't", 'how', 'during', 'having', 'a', 'if', 'i', 'own', 'which', 'again', 'further', 'after', 'wasn', 'his', 'don', 'my', 'an', 'while', "mightn't", 'down', 'y', 'in', "should've", 'between', 'above', "didn't", 'yours', 'she', 'her', 'your', 'too', "weren't", 'you', 'were', 'has', 'to', 'do', 'through', "it's", 'ours', 'hers', "hadn't", 'then', 'mightn', 'yourself', 'only', "wasn't", 'our', 'they', 'not', 'some', "shouldn't", 'who', 'against', 'over', 'no', "she's", 'am', 'it', 'of', 'so', 'ourselves', "needn't", 'same', 'does', 'ain', 'any', 'is', 'will', 'doing', 'until', 'under', 'there', "couldn't", 'theirs', 'hasn', 'but', "you'd", 'the', 'before', 'we', 't', 'being', "you're", 'by', 'here', 'm', 'its', 'needn', "won't", "you'll", 'that', 'just', "that'll", 'be', 'off', "isn't", 'whom', 'themselves', 'myself', 'what', 'him', 'than', "aren't", 'or', 'this', 'and', 'wouldn', 'now', 'weren', 's', 'me', 'been', "doesn't", 're', 'did', 've', 'below', 'yourselves', 'at', 'on', 'very', 'doesn', 'he', 'herself', 'from', 'himself', 'shan', 'few', 'all', 'ma', 'those', "wouldn't", 'their', "hasn't", "haven't", 'why', "shan't", 'these', 'o', 'couldn', 'itself', 'into', 'as', 'each', 'had', 'with', 'aren', 'for', 'both', 'should', "mustn't", 'd', 'won', 'out', 'where', "you've", 'up', 'shouldn', 'have', 'because', 'when', 'was', 'nor', 'are', 'll', 'didn', 'other'}
    #filtrera på max 3 chars.
    
    example_sent = "ProOptical He'l\"l0 and hÄéj! is . & it's tov its both above ProOPtical"
    print(tokenize(example_sent))
    #tokenize_name("Hello", [token_to_lower])

if __name__ == "__main__":
    main()