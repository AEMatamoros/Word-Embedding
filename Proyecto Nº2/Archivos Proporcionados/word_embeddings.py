from os import listdir, getcwd
from os.path import isfile, join
from gensim.models import Word2Vec
import numpy as np
from unicodedata import normalize
import re
import math


class WordEmbedding:

    def __init__(self, corpus_path):
        self._corpus = []
        document = list(listdir(corpus_path))

        for doc in document:
            openDoc = open(join(corpus_path, doc), encoding='utf-8')
            text = openDoc.read().lower()

            trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
            text = normalize("NFKC", normalize("NFKD", text).translate(trans_tab))

            text_Character = re.sub("[^a-zA-Z0-9 \n\.]", " ", text).replace(".","")

            text_format = re.sub(r'\d', '', text_Character)

            list_word = text_format.split()

            self._corpus.append(list_word)

        # print(self._corpus)

    def bag_of_words(self):
        prebag = []
        for sentence in self._corpus:
            for word in sentence:
                if word not in prebag:
                    prebag.append(word)

        self.n = len(prebag)
        sortedWords = sorted(prebag)
        bag_of_words = np.zeros(shape=(self.n, len(self._corpus)))

        i = 0
        j = 0

        for sentence in self._corpus:
            for sortWord in sortedWords:
                for word in sentence:
                    if word == sortWord and i < self.n:
                        bag_of_words[i][j] += 1
                i += 1
            j += 1
            i = 0

        return (bag_of_words)

    def tf_idf(self):
        # Se obtiene la frecuencia de cada palabra
        doneWords = []  # Almacena las palabras que ya fueron analizadas
        # doneWordsListFreq = []#Almacena las frequiencias por documento
        count = 0  # Cuenta el numero de repeticiones por palabra
        numberOfDocs = 0
        numberOfWords = []
        for wordsList in self._corpus:  # Analisis por lista de palabras
            numberOfDocs += 1
            wordsPerDoc = 0
            for word in wordsList:  # Analisis por palabra
                wordsPerDoc += 1
                if word not in doneWords:
                    doneWords.append(word)  # Agregar Palabras para analisis
                count = 0
            numberOfWords.append(wordsPerDoc)
        # Creando arreglo
        freqsMatriz = np.zeros(shape=(self.n, len(self._corpus)))
        # llenar Matriz
        # print(doneWords)
        # Matriz de frecuencia de la palabras en los documentos
        freqsMatriz = self.bag_of_words()
        # cantidad de documentos que contiene la palabra
        # print(freqsMatriz)
        wordCont = []
        count = 0
        doneWords.sort()
        for word in doneWords:
            for sentences in self._corpus:
                readyWords = []
                for sentence in sentences:
                    if sentence == word:
                        if word not in readyWords:
                            readyWords.append(word)
                            count += 1

            wordCont.append(count)
            count = 0

        # tf:
        # print(wordCont)
        # print(freqsMatriz)

        tf = np.zeros(shape=(self.n, len(self._corpus)))
        for i in range(self.n):
            for j in range(len(self._corpus)):
                tf[i][j] = (freqsMatriz[i][j]/numberOfWords[j])

        # idf:
        idf = np.zeros(shape=(self.n, len(self._corpus)))
        for i in range(self.n):
            for j in range(len(self._corpus)):
                tt = math.log10(len(self._corpus)/wordCont[i])
                idf[i][j] = tt

        # itf_idf:
        tf_idf = np.zeros(shape=(self.n, len(self._corpus)))
        for i in range(self.n):
            for j in range(len(self._corpus)):
                tf_idf[i][j] = tf[i][j] * idf[i][j]
        # print(freqsMatriz)
        # print(tf)
        # print(idf)
        # print(tf_idf)
        return (tf_idf)

    def word2vec(self):
        documents = self._corpus
        vecs = []

        for doc in documents:
            model = Word2Vec(doc, size=50, min_count=1, workers=1, seed=1)
            vecs.append(list(model.wv.vocab))
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'll', 'm',
                   'n', 'ñ', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w','x', 'y', 'z']
        values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

        abc = dict(zip(letters, values))
        w2v = dict()

        for elem in vecs:
            for word in elem:
                if(word in abc):
                    w2v[word] = abc[word]

        








nuevaruta = str(getcwd())+r'/test_corpus'

word = WordEmbedding(nuevaruta)

word.bag_of_words()

word.tf_idf()

word.word2vec()
