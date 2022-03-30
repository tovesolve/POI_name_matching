from cmath import cos
from bpemb import BPEmb
#from token_based_func import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cosine
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# bpEmb consists of BPE+Glove.
# tokenizes and encodes the strings using Byte-Pair-Encoding (BPE) and embeds it using Glove.
def bpEmb(word, dim=300, vs=50000):
    bpemb_en = BPEmb(lang="en", dim=dim, vs=vs) 
    
    # this step is not needed as the encoding is done in embed    
    # tokens = bpemb_en.encode(word)
    # print(tokens)

    embedding = bpemb_en.embed(word)
    embedding_mean = embedding.mean(axis=0).reshape(1,-1)
    return embedding_mean

# sBert (sentenceBERT): sentence-transformer (sentence embedding)
def sBERT(word):
    # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # model.save("all-mpnet-base-v2")
    model = SentenceTransformer("all-mpnet-base-v2")
    embedding_mean = model.encode(word).reshape(1,-1) #mean poolin is calculated in encode
    return embedding_mean

# Bert: Fill-Mask (mean of word-embeddings)
def BERT(word):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    encoded_input = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    embedding_mean = mean_pooling(model_output, encoded_input['attention_mask'])
    embedding = F.normalize(embedding_mean, p=2, dim=1).reshape(1,-1)
    return embedding
    
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
def main():
    #s1 = sBERT("yellow")
    #s2 = sBERT("black")
    s1 = BERT("yellow")
    s2 = BERT("black")
    #s1 = bpEmb("yellow")
    #s2 = bpEmb("black")
    print(cosine(s1, s2))

if __name__ == "__main__":
    main()
