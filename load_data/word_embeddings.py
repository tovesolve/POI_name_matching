#import gensim.downloader as api
import numpy as np
import torch.nn.functional as F
from bpemb import BPEmb as BPEmbedding
from cmath import cos
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from tokenizer import tokenize_name
from transformers import BertTokenizer, BertModel
import torch
from usif import *


#===================================BPEmb======================
# bpEmb consists of BPE+Glove.
# tokenizes and encodes the strings using Byte-Pair-Encoding (BPE) and embeds it using Glove.
def get_embedding_BPEmb(word, model):
    '''
    Creates an embedding for a POI using a pretrained BPEmb (BPE+glove) model.
    The model can be loaded with, for example:
        model_BPEmb = BPEmb(lang="en", dim=300, vs=50000) 
    '''
    # this step is not needed as the encoding is done in embed    
    # tokens = bpemb_en.encode(word)
    # print(tokens)

    np.set_printoptions(threshold=sys.maxsize)
    embedding = model.embed(word)
    # print("word: ", word)
    # print("embedding before mean/smooting: ", embedding)
    
    embedding_mean = embedding.mean(axis=0).reshape(1,-1)
    #print("embedding after old mean: ", embedding_mean)
    return embedding_mean
    
#===================================sBERT=======================
# sBert (sentenceBERT): sentence-transformer
def get_embedding_sBERT(word, model):
    '''
    Creates an embedding for a POI using a pretrained sBERT model.
    The model can be loaded with sentence transformers, for example:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    '''
    embedding_mean = model.encode(word).reshape(1,-1) #mean pooling is calculated in encode
    return embedding_mean


#===================================BERT=======================
# Bert: Fill-Mask (mean of word-embeddings)
def get_embedding_BERT(word, model, tokenizer):
    '''
    Creates an embedding for a POI using a pretrained sBERT model.
    The model can be loaded with sentence transformers, for example:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    '''
    encoded_input = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding_mean_pool = mean_pooling(model_output, encoded_input['attention_mask'])
    return embedding_mean_pool[0].reshape(1,-1)
    
#Mean Pooling - Take attention mask into account for correct averaging, helper for BERT
def mean_pooling(model_output, attention_mask):
    '''
    Take attention mask into account for correct averaging, helper for BERT
    '''
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#===================================Word2vec=====================
def get_embedding_word2vec(poi_name, model):
    '''
    Creates an embedding for a POI using a pretrained word2vec model.
    The model can be loaded from genism using for example:
        api.load("word2vec-google-news-300")
    '''
    poi_tokenized = tokenize_name(poi_name)     #Tokenize poi_name, create a tokenized list

    #Embed each token in the tokenlist. Embedded vectors is added to the poi_tokenvectors list
    poi_tokenvectors = []
    for token in poi_tokenized:
        vec = model.get_vector(token)#.tolist()
        poi_tokenvectors.append(vec)

    poi_tokenvectors=np.array([np.array(xi) for xi in poi_tokenvectors])    #Convert vector list to np.ndarray-format

    poi_vector_mean = poi_tokenvectors.mean(axis=0)     #Create one vector embedding for the poi_name out of vectors for each token

    return poi_vector_mean


#===================================Gensim=====================
def get_embedding_glove(poi_name, model):
    '''
    Creates an embedding for a POI using a pretrained glove model.
    The model can be loaded from genism using for example:
        api.load("glove-twitter-200")
    '''

    poi_tokenized = tokenize_name(poi_name)     #Tokenize poi_name, create a tokenized list

    #Embed each token in the tokenlist. Embedded vectors is added to the poi_tokenvectors list 
    poi_tokenvectors = []
    for token in poi_tokenized:
        vec = model.get_vector(token)#.tolist()
        poi_tokenvectors.append(vec)

    poi_tokenvectors=np.array([np.array(xi) for xi in poi_tokenvectors])     #Convert vector list to np.ndarray-format

    poi_vector_mean = poi_tokenvectors.mean(axis=0)     #Create one vector embedding for the poi_name out of vectors for each token

    return poi_vector_mean

#===================================Fasttext=====================
def get_embedding_fasttext(poi_name, model):
    '''
    Creates an embedding for a POI using a pretrained fasttext model.
    The model can be loaded from genism using for example:
        model_fasttext = api.load("fasttext-wiki-news-subwords-300")
    '''

    poi_tokenized = tokenize_name(poi_name)     #Tokenize poi_name, create a tokenized list

    #Embed each token in the tokenlist. Embedded vectors is added to the poi_tokenvectors list 
    poi_tokenvectors = []
    for token in poi_tokenized:
        vec = model.get_vector(token)#.tolist()
        poi_tokenvectors.append(vec)

    poi_tokenvectors=np.array([np.array(xi) for xi in poi_tokenvectors])     #Convert vector list to np.ndarray-format

    poi_vector_mean = poi_tokenvectors.mean(axis=0)     #Create one vector embedding for the poi_name out of vectors for each token

    return poi_vector_mean

# def get_sif_embedding(word):
#     sif_embedding = get_paranmt_usif()
#     #print("embedding with SIF:", sif_embedding.embed(word))
#     return sif_embedding.embed(word)



def main():
    #poi1 = "Jewett Orthopedic Clinic"
    #poi2 = "Jewett Orthopedic Convenient Care Center"
    #print("Compared POIs: ", poi1, " and ", poi2)
    

    #BPEmb
    dim=300
    vs=50000
    model_BPEmb = BPEmbedding(lang="en", dim=dim, vs=vs) 
    
    name1 = "champlain Square"
    name2 = "champlain Mall"
    
    emb1_BPEmb = get_embedding_BPEmb(name1, model_BPEmb)
    emb2_BPEmb = get_embedding_BPEmb(name2, model_BPEmb)
    print("cosine: ", sklearn_cosine_similarity(emb1_BPEmb, emb2_BPEmb))
    # print("Cosine similarity score using BPEmb model: ", sklearn_cosine_similarity(emb1_BPEmb, emb2_BPEmb))
    
    # p1 = get_sif_embedding([name1])
    # p2 = get_sif_embedding([name2])
    #p2 = get_sif_embedding("Jewett Orthopedic Convenient Care Center")
    #print("p1 embedding: ", p1)
    
    #print("p2 embedding: ", p2)
    # print("cosine: ", sklearn_cosine_similarity(p1, p2))

    # #sBERT
    # # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # # model.save("all-mpnet-base-v2")
    # model_sBERT = SentenceTransformer("all-mpnet-base-v2")
    # emb1_sBERT = get_embedding_sBERT(poi1, model_sBERT)
    # emb2_sBERT = get_embedding_sBERT(poi2, model_sBERT)
    # print("Cosine similarity score using sBERT model: ", sklearn_cosine_similarity(emb1_sBERT, emb2_sBERT))
    
    # #BERT
    # tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    # model_BERT = BertModel.from_pretrained("bert-base-uncased")
    # emb1_BERT = get_embedding_BERT(poi1, model_BERT, tokenizer_BERT)
    # emb2_BERT = get_embedding_BERT(poi2, model_BERT, tokenizer_BERT)
    # print("Cosine similarity score using BERT model: ", sklearn_cosine_similarity(emb1_BERT, emb2_BERT))

    # #Word2vec
    # #model_word2vec = api.load("word2vec-google-news-300")  # load pre-trained word-vectors from gensim-data
    # #model_word2vec.save('vectors_word2vec.kv')
    # model_word2vec = KeyedVectors.load('vectors_word2vec.kv')
    # emb1_word2vec = get_embedding_word2vec(poi1, model_word2vec)
    # emb2_word2vec = get_embedding_word2vec(poi2, model_word2vec)
    # print("Cosine similarity score using Word2vec model: ", sklearn_cosine_similarity(emb1_word2vec.reshape(1,-1), emb2_word2vec.reshape(1,-1)))

    # #GloVe
    # #model_glove = api.load('glove-wiki-gigaword-200')  # load pre-trained word-vectors from gensim-data
    # #model_glove.save('vectors.kv')
    # model_glove = KeyedVectors.load('vectors.kv')
    # emb1_glove = get_embedding_glove(poi1, model_glove)
    # emb2_glove = get_embedding_glove(poi2, model_glove)
    # print("Cosine similarity score using glove model: ", sklearn_cosine_similarity(emb1_glove.reshape(1,-1), emb2_glove.reshape(1,-1)))

    # #Fasttext
    # #model_fasttext = api.load("fasttext-wiki-news-subwords-300")  # load pre-trained word-vectors from gensim-data
    # #model_fasttext.save('vectors_fasttext.kv')
    # model_fasttext = KeyedVectors.load('vectors_fasttext.kv')
    # emb1_fasttext = get_embedding_fasttext(poi1, model_fasttext)
    # emb2_fasttext = get_embedding_fasttext(poi2, model_fasttext)
    # print("Cosine similarity score using fasttext model: ", sklearn_cosine_similarity(emb1_fasttext.reshape(1,-1), emb2_fasttext.reshape(1,-1)))

if __name__ == "__main__":
    main()
