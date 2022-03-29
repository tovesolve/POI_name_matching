from cmath import cos

from sklearn.datasets import make_biclusters
from bpemb import BPEmb
from token_based_func import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
import OpenTextbot.src.Algebra as Algebra 

bpemb_en = BPEmb(lang="en", dim=300, vs=50000)
s1 = "New York" #"Boston Common Coffee Co."
s2 = "Tiva" #"Boston Bean Stock Coffee"


Tokens1 = bpemb_en.encode(s1)
Tokens2 = bpemb_en.encode(s2)
print(Tokens1)
print(Tokens2)

embedding1 = bpemb_en.embed(s1)
embedding2 = bpemb_en.embed(s2)
print(embedding1)
print(embedding1.shape)
print(embedding2.shape)
m_emb = embedding1.mean(axis=0)
m_emb2 = embedding2.mean(axis=0)
print(m_emb)
print(m_emb.reshape(1,-1))
print(sklearn_cosine_similarity(m_emb.reshape(1,-1), m_emb2.reshape(1,-1)))
