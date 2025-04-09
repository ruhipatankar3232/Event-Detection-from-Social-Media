import pandas as pd
import nltk
import numpy as np
import heapq

## Finding TF-IDF for context
def TF_IDF(corpus):
    # Remove all the special characters and multiply with empty spaces
    #TFIDF_Doc = []
    #data = np.load('Data.npy', allow_pickle=True)
    #for i in range(len(data)):
    #corpus = data[0][0]
    #corpus = corpus.tolist()
    # corpus = emailspams[0]
    wordfreq = {}
    for sentence in corpus:
        #print(sentence)
        if type(sentence) == float:
            ans = 0
        else:
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                if token not in wordfreq.keys():
                    wordfreq[token] = 1  # a dictionary of word frequencies are created
                else:
                    wordfreq[token] += 1

    most_freq = heapq.nlargest(200, wordfreq,
                               key=wordfreq.get)  # filtered the top 200 most frequently occurred words

    # determine the IDF values
    word_idf_values = {}
    for token in most_freq:
        doc_containing_word = 0
        for document in corpus:
            #print(document)
            if type(document) == float:
                ans = 0
            else:
                if token in nltk.word_tokenize(document):
                    doc_containing_word += 1
        word_idf_values[token] = np.log(len(corpus) / (1 + doc_containing_word))

    ## determine TF values
    word_tf_values = {}
    for token in most_freq:
        sent_tf_vector = []
        for document in corpus:
            #print(document)
            if type(document) == float:
                ans = 0
            else:
                doc_freq = 0
                for word in nltk.word_tokenize(document):
                    if token == word:
                        doc_freq += 1
                try:
                    word_tf = doc_freq / len(nltk.word_tokenize(document))
                except ZeroDivisionError:
                    word_tf = 0.0

            sent_tf_vector.append(word_tf)
        word_tf_values[token] = sent_tf_vector

    # Determining TF-IDF
    tfidf_values = []
    for token in word_tf_values.keys():
        tfidf_sentences = []
        for tf_sentence in word_tf_values[token]:
            tf_idf_score = tf_sentence * word_idf_values[token]
            tfidf_sentences.append(tf_idf_score)
        tfidf_values.append(tfidf_sentences)

    return tfidf_values

