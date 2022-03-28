from nltk.corpus import wordnet
import nltk
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
embeddings = model.encode(sentences)
print(embeddings)
#from sematch.evaluation import WordSimEvaluation
#from sematch.semantic.similarity 
#import sematch #  WordNetSimilarity

#nltk.download('wordnet')
#nltk.download('omw-1.4')

#wns = sematch.semantic.similarity.WordNetSimilarity()
#print(wns.word_similarity("dog", "cat", "li"))