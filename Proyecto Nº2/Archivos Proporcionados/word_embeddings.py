from os import listdir, getcwd
from os.path import isfile, join
from gensim import corpora
from gensim.models import Word2Vec, TfidfModel
from gensim.utils import simple_preprocess
import numpy as np
from unicodedata import normalize
import re 


class WordEmbedding:

  def __init__(self, corpus_path):
    self._corpus = []
    document = list(listdir(corpus_path))

    for doc in document:
      openDoc = open(join(corpus_path, doc) , encoding='utf-8')
      text = openDoc.read().lower()

      trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
      text = normalize("NFKC", normalize("NFKD", text).translate(trans_tab))

      text_Character = re.sub("[^a-zA-Z0-9 \n\.]", " ", text).replace("."," ")

      text_format = re.sub(r'\d','',text_Character)

      list_word = text_format.split()

      self._corpus.append(list_word)
        
    print(self._corpus)


  def bag_of_words(self):
    prebag = []
    for sentence in self._corpus:
      for word in sentence: 
        if word not in prebag:
          prebag.append(word)
    
          
    self.n = len(prebag)
    sortedWords = sorted(prebag)
    bag_of_words = np.zeros(shape=(self.n,len(self._corpus)))
   
    i = 0
    j = 0
    
    for sentence in self._corpus:
      for sortWord in sortedWords:
        for word in sentence:
          if word == sortWord and i<self.n:
            bag_of_words[i][j]+=1
        i+=1
      j+=1
      i=0
    
    return (bag_of_words)

  def tf_idf(self):
    documents = self._corpus
    print(documents)
    """documents = [d.split() for d in documents]
    dictionary = corpora.Dictionary(documents)
    
    for line in documents:
      tokenized_list = simple_preprocess(line, deacc=True)
      corpus = [dictionary.doc2bow(tokenized_list, allow_update=True)]"""


  def word2vec(self):
    pass

  
nuevaruta = str(getcwd())+r'/test_corpus'

word = WordEmbedding(nuevaruta)

word.bag_of_words()

word.tf_idf()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
