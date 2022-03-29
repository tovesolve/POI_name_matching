from cmath import cos
from bpemb import BPEmb
#from token_based_func import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity as cosine

# bpEmb consists of BPE+Glove.
# tokenizes and encodes the strings using Byte-Pair-Encoding (BPE) and embeds it using Glove.
def bpEmb(word, dim=300, vs=50000):
    bpemb_en = BPEmb(lang="en", dim=dim, vs=vs) 
    
    # this step is not needed as the encoding is done in embed    
    # tokens = bpemb_en.encode(word)
    # print(tokens)

    embedding = bpemb_en.embed(word)
    embedding_mean = embedding.mean(axis=0).reshape(1,-1)
    #print(embedding_mean)
    return embedding_mean

# sBert (sentenceBERT): sentence-transformer
def sBERT(word):
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model.save("all-mpnet-base-v2")
    model = SentenceTransformer("all-mpnet-base-v2")
    embedding_mean = model.encode(word).reshape(1,-1) #mean poolin is calculated in encode
    #print(embedding_mean)
    return embedding_mean
    
def main():
    s1 = sBERT("yellow")
    s2 = sBERT("black")
    #s1 = bpEmb("yellow")
    #s2 = bpEmb("black")
    print(cosine(s1, s2))

if __name__ == "__main__":
    main()
