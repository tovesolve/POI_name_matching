import gensim.downloader as api
import numpy as np
import torch.nn.functional as F
from bpemb import BPEmb
from cmath import cos
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from tokenizer import tokenize_name

#===================================BPEmb======================
# bpEmb consists of BPE+Glove.
# tokenizes and encodes the strings using Byte-Pair-Encoding (BPE) and embeds it using Glove.
def BPEmb_embedding(word, model):
    '''
    Creates an embedding for a POI using a pretrained BPEmb (BPE+glove) model.
    The model can be loaded with, for example:
        model_BPEmb = BPEmb(lang="en", dim=300, vs=50000) 
    '''
    # this step is not needed as the encoding is done in embed    
    # tokens = bpemb_en.encode(word)
    # print(tokens)

    embedding = model.embed(word)
    embedding_mean = embedding.mean(axis=0).reshape(1,-1)
    #print(embedding_mean)
    return embedding_mean

#===================================sBERT=======================
# sBert (sentenceBERT): sentence-transformer
def sBERT(word, model):
    '''
    Creates an embedding for a POI using a pretrained sBERT model.
    The model can be loaded with sentence transformers, for example:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    '''
    embedding_mean = model.encode(word).reshape(1,-1) #mean pooling is calculated in encode
    #print(embedding_mean)
    return embedding_mean

#===================================Word2vec=====================
def word2vec_embedding(poi_name, model):
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
def glove_embedding(poi_name, model):
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
def fasttext_embedding(poi_name, model):
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



def main():
    poi1 = "Jewett Orthopedic Clinic"
    poi2 = "Jewett Orthopedic Convenient Care Center"
    print("Compared POIs: ", poi1, " and ", poi2)

    #BPEmb
    dim=300
    vs=50000
    model_BPEmb = BPEmb(lang="en", dim=dim, vs=vs) 
    emb1_BPEmb = BPEmb_embedding(poi1, model_BPEmb)
    emb2_BPEmb = BPEmb_embedding(poi2, model_BPEmb)
    print("Cosine similarity score using BPEmb model: ", sklearn_cosine_similarity(emb1_BPEmb, emb2_BPEmb))

    #sBERT
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model.save("all-mpnet-base-v2")
    model_sBERT = SentenceTransformer("all-mpnet-base-v2")
    emb1_sBERT = sBERT(poi1, model_sBERT)
    emb2_sBERT = sBERT(poi2, model_sBERT)
    print("Cosine similarity score using sBERT model: ", sklearn_cosine_similarity(emb1_sBERT, emb2_sBERT))

    #Word2vec
    #model_word2vec = api.load("word2vec-google-news-300")  # load pre-trained word-vectors from gensim-data
    #model_word2vec.save('vectors_word2vec.kv')
    model_word2vec = KeyedVectors.load('vectors_word2vec.kv')
    emb1_word2vec = word2vec_embedding(poi1, model_word2vec)
    emb2_word2vec = word2vec_embedding(poi2, model_word2vec)
    print("Cosine similarity score using Word2vec model: ", sklearn_cosine_similarity(emb1_word2vec.reshape(1,-1), emb2_word2vec.reshape(1,-1)))

    #GloVe
    #model_glove = api.load('glove-wiki-gigaword-200')  # load pre-trained word-vectors from gensim-data
    #model_glove.save('vectors.kv')
    model_glove = KeyedVectors.load('vectors.kv')
    emb1_glove = glove_embedding(poi1, model_glove)
    emb2_glove = glove_embedding(poi2, model_glove)
    print("Cosine similarity score using glove model: ", sklearn_cosine_similarity(emb1_glove.reshape(1,-1), emb2_glove.reshape(1,-1)))

    #Fasttext
    #model_fasttext = api.load("fasttext-wiki-news-subwords-300")  # load pre-trained word-vectors from gensim-data
    #model_fasttext.save('vectors_fasttext.kv')
    model_fasttext = KeyedVectors.load('vectors_fasttext.kv')
    emb1_fasttext = fasttext_embedding(poi1, model_fasttext)
    emb2_fasttext = fasttext_embedding(poi2, model_fasttext)
    print("Cosine similarity score using fasttext model: ", sklearn_cosine_similarity(emb1_fasttext.reshape(1,-1), emb2_fasttext.reshape(1,-1)))

if __name__ == "__main__":
    main()
