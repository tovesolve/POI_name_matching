from cmath import cos
from bpemb import BPEmb
from token_based_func import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import numpy as np
import OpenTextbot.src.Algebra as Algebra 
import gensim.downloader as api
from gensim.models import KeyedVectors


#===================================BPEmb=====================
bpemb_en = BPEmb(lang="en", dim=300, vs=50000)
s1 = 'bean boston'#"Boston Common Coffee Co."
s2 = 'Jewett Orthopedic Clinic'#"Boston Bean Stock Coffee"


Tokens1 = bpemb_en.encode(s1)
Tokens2 = bpemb_en.encode(s2)
print(Tokens1)
print(Tokens2)

embedding1 = bpemb_en.embed(s1)
embedding2 = bpemb_en.embed(s2)
print(embedding1.shape)
print(embedding2.shape)
m_emb = embedding1.mean(axis=0)
m_emb2 = embedding2.mean(axis=0)
#print(m_emb)
#print(m_emb.reshape(1,-1))
print(sklearn_cosine_similarity(m_emb.reshape(1,-1), m_emb2.reshape(1,-1)))



#===================================Word2vec=====================

def word2vec_embedding(poi_tokenized, model):
    '''
    Dependencies:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
    '''
    #Embed each token in the tokenlist. Embedded vectors is added to the poi_tokenvectors list
    poi_tokenvectors = []
    for token in poi_tokenized:
        vec = model.get_vector(token)#.tolist()
        poi_tokenvectors.append(vec)

    #Convert vector list to np.ndarray-format
    poi_tokenvectors=np.array([np.array(xi) for xi in poi_tokenvectors])

    poi_vector_mean = poi_tokenvectors.mean(axis=0)

    return poi_vector_mean

#word_vectors_word2vec = api.load("word2vec-google-news-300")  # load pre-trained word-vectors from gensim-dat
#word_vectors_word2vec.save('vectors_word2vec.kv')
word_vectors_word2vec = KeyedVectors.load('vectors_word2vec.kv')
emb1 = word2vec_embedding(s1.lower().split(), word_vectors_word2vec)
emb2 = word2vec_embedding(s2.lower().split(), word_vectors_word2vec)

print("Cosine Word2vec sim with function: ", sklearn_cosine_similarity(emb1.reshape(1,-1), emb2.reshape(1,-1)))



#===================================Gensim=====================

def glove_embedding(poi_tokenized, model):
    '''
    Dependencies:
    import gensim.downloader as api
    from gensim.models import KeyedVectors
    '''
    #Embed each token in the tokenlist. Embedded vectors is added to the poi_tokenvectors list
    poi_tokenvectors = []
    for token in poi_tokenized:
        vec = model.get_vector(token)#.tolist()
        poi_tokenvectors.append(vec)

    #Convert vector list to np.ndarray-format
    poi_tokenvectors=np.array([np.array(xi) for xi in poi_tokenvectors])

    poi_vector_mean = poi_tokenvectors.mean(axis=0)

    return poi_vector_mean



#word_vectors = api.load("glove-twitter-200")  # load pre-trained word-vectors from gensim-dat
#word_vectors.save('vectors.kv')
word_vectors = KeyedVectors.load('vectors.kv')

# similarity = word_vectors.similarity(s1, s2)
# print("similarity: ", similarity)

# similarity2 = word_vectors.most_similar(s1)
# print(similarity2)
# print(f"{similarity:.4f}")

# sentence_obama = 'Boston Common Coffee Co.'.lower().split()
# sentence_president = 'Boston Bean Stock Coffee'.lower().split()
# similarity = word_vectors.n_similarity(sentence_obama, sentence_president)
# print("result1: ", f"{similarity:.4f}")

sentence_obama = s1.lower().split()
sentence_president = s2.lower().split() #onvenient Care Center

s1_mean = glove_embedding(sentence_obama, word_vectors)
s2_mean = glove_embedding(sentence_president, word_vectors)

print("Cosine sim with function: ", sklearn_cosine_similarity(s1_mean.reshape(1,-1), s2_mean.reshape(1,-1)))

combined_words_s1 = []
for word in sentence_obama:
    print(word, ":")
    vec = word_vectors.get_vector(word).tolist()
    combined_words_s1.append(vec)
#print("comb words s1: ", combined_words_s1)
#print(combined_words_s1.shape)

combined_words_s2 = []
for word in sentence_president:
    print(word, ":")
    vec = word_vectors.get_vector(word).tolist()
    combined_words_s2.append(vec)
#print("comb words s2: ", combined_words_s2)

combined_words_s1=np.array([np.array(xi) for xi in combined_words_s1])
combined_words_s2=np.array([np.array(xi) for xi in combined_words_s2])

print("shape combined_words s1: ", combined_words_s1.shape)
print("shape combined_words s2: ", combined_words_s2.shape)


comb1_mean = combined_words_s1.mean(axis=0)
comb2_mean = combined_words_s2.mean(axis=0)
print("comb1_mean shape: ", comb1_mean.shape)
print("Cosine sim: ", sklearn_cosine_similarity(comb1_mean.reshape(1,-1), comb2_mean.reshape(1,-1)))


#def glove_embedding(tokenized_poi1, tokenized_poi2, model)   # poi1: ["Toves", "Cafe"], poi2: ["Lucys", "hotell"]
# model = Load embedding
# do embed seperatly for poi1 and poi2
    #poi1_embedding
    #poi2_embedding

# create sentence mean
# return sentence mean1, sentence mean2
