import json
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from numpy import matlib
from GOA import GOA
from Global_Vars import Global_Vars
from MAO import MAO
from Model_LSTM import Model_LSTM
from Model_RNN import Model_RNN
from Model_Resnet import Model_Resnet
import gensim
from gensim.models import word2vec
from sklearn.decomposition import PCA
from Model_TransResBiLSTM import Model_TransResBiLSTM
from Objective_Function import objfun
from PFOA import PFOA
from Plot_Results import *
from Proposed import Proposed
from SGO import SGO
from Tfidf import TF_IDF


# Removing punctuations
def rem_punct(my_str):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct += char
    return no_punct


# Dataset
an = 0
if an == 1:
    directory = './Dataset/Dataset1/'
    dir1 = os.listdir(directory)
    texts = []
    targ = []
    for i in range(len(dir1)):
        file = directory + dir1[i]
        rr = open(directory + dir1[i])
        read = json.load(rr)
        for j in range(35):
            read1 = read['user_' + str(j + 1)]['tweets']
            for i in range(len(read1)):
                text = read['user_' + str(j + 1)]['tweets'][i]['text']
                read2 = read['user_' + str(j + 1)]['tweets'][i]['label']
                texts.append(text)
                targ.append(read2)
    Targ = np.asarray(targ)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    texts = np.asarray(texts)
    np.save('Data.npy', texts)
    np.save('Targets.npy', tar)

# Pre-Processing
an = 0
if an == 1:
    Feats = []
    Data = np.load('Data.npy', allow_pickle=True)
    Feat = []
    for i in range(len(Data)):
        print(i)
        D = Data[i]
        if type(D) == float:
            D = 'No Statement'
        ps = PorterStemmer()
        punc = rem_punct(D)
        text_tokens = word_tokenize(punc)  # convert in to tokens
        stem = []
        for w in text_tokens:  # Stemming
            stem_tokens = ps.stem(w)
            stem.append(stem_tokens)
        words = [word for word in stem if
                 not word in stopwords.words()]  # tokens without stop words
        # Punctuation Removal
        prep = rem_punct(words)
        dat = []
        v = []
        for m in sent_tokenize(str(prep)):
            temp1 = []
            # tokenize the sentence into words
            for n in word_tokenize(m):
                temp1.append(n.lower())
            dat.append(temp1[0])
        Feat.append(dat)
    np.save('Preprocessed_Data.npy', Feat)

# Feature Extraction - TFIDF
an = 0
if an == 1:
    Data = np.load('Preprocessed_Data.npy', allow_pickle=True)
    pca = PCA(n_components=1)
    Vector = []
    for d in range(len(Data)):  # len(Data)
        if not len(Data[d]):
            data1 = ['Word']
        else:
            data1 = np.asarray(Data[d])[0]
        vect = np.zeros((1, 101))  # len(data1)
        for i in range(1):  # len(data1)
            print(d, i)
            if not len(data1[i]):
                vect[i, :] = np.zeros((101))
            else:
                val = data1[i]
                model2 = gensim.models.Word2Vec(val, min_count=1,
                                                window=5, sg=1)  # Word2vector
                v = model2.wv.vectors
                p1 = pca.fit_transform(v.transpose())
                vect[i, 0:100] = p1[0:100].reshape(1, -1)
                vect[i, 100] = np.max(np.unique(TF_IDF(val)))
        Vector.append(vect.reshape(-1))
    np.save('Vector.npy', Vector)


# weight optimization
an = 0
if an == 1:
    Feat = np.load('Vector.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    Global_Vars.Feat = Feat
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat(0.01 * np.ones(Chlen), Npop, 1)
    xmax = matlib.repmat(0.99 * np.ones(Chlen), Npop, 1)
    fname = objfun
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("SGO...")
    [bestfit1, fitness1, bestsol1, time1] = SGO(initsol, fname, xmin, xmax, Max_iter)  # SGO

    print("MAO...")
    [bestfit2, fitness2, bestsol2, time2] = MAO(initsol, fname, xmin, xmax, Max_iter)  # MAO

    print("GOA...")
    [bestfit4, fitness4, bestsol4, time3] = GOA(initsol, fname, xmin, xmax, Max_iter)  # GOA

    print("PFOA...")
    [bestfit3, fitness3, bestsol3, time4] = PFOA(initsol, fname, xmin, xmax, Max_iter)  # PFOA

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol.npy', BestSol)

# Classification
an = 0
if an == 1:
    Selected_Features = np.load('Preprocessed_Data.npy', allow_pickle=True)
    Target = np.load('Targets.npy', allow_pickle=True)
    BestSol = np.load('BestSol.npy', allow_pickle=True)
    K = 5
    Per = 1 / 5
    Perc = round(Selected_Features.shape[0] * Per)
    Eval_all = []
    for i in range(K):
        Train_Data = Selected_Features[:Perc, :]
        Train_Target = Target[:Perc, :]
        Test_Data = Selected_Features[Perc:, :]
        Test_Target = Target[Perc:, :]
        Eval = np.zeros((10, 14))
        for j in range(BestSol.shape[1]):
            print(i, j)
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :] = Model_TransResBiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval[0, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[1, :], pred = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[2, :], pred = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[3, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[4, :], pred = Model_TransResBiLSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval_all.append(Eval)
    np.save('Eval_all.npy', Eval_all)


plot_results()
Plot_ROC_Curve()