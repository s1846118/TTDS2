from nltk.util import transitive_closure
from numpy import genfromtxt
import numpy as np
import re
import pandas as pd
from nltk import text
from nltk.stem import PorterStemmer
import math
import csv
from pandas.io.parsers import read_csv
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import cartesian
from sklearn import svm
import scipy



class eval():

    def __init__(self, path, path2):

        self.sys_np = genfromtxt(path, delimiter=',')[1:]
        self.q_np = genfromtxt(path2, delimiter=',')[1:]

        self.systems = int(np.max(self.sys_np[:,0]))
        self.queries = int(np.max(self.sys_np[:,1]))

    def create_index(self):
        self.index = {} # Each system will be a key

        for system in range(self.systems):
            self.index[system+1] = {} # Each query will be a key
            for query in range(self.queries):
                self.index[system+1][query+1] = {'Pa10':0,'Ra50':0,'r-precision':0,'AP':0,'nDCGa10':0,'nDCGa20':0} # Instance for each query per system

    def stats(self):
        for system in range(self.systems):
            for query in range(self.queries):
                relQ = self.q_np[self.q_np[:,0] == query+1,:]
                # First x results for each system for each query
                results = self.sys_np[self.sys_np[:,0] == system+1,:]
                results10 = results[results[:,1] == query+1,:][:10,:]
                results20 = results[results[:,1] == query+1,:][:20,:]
                results50 = results[results[:,1] == query+1,:][:50,:]
                resultsR = results[results[:,1] == query+1,:][:len(relQ),:]
                resultsAvg = results[results[:,1] == query+1,:]

                # Relevant and retreived
                RaR10 = len(np.intersect1d(results10[:,2], relQ[:,1])) 
                RaR50 = len(np.intersect1d(results50[:,2], relQ[:,1]))
                RaRR = len(np.intersect1d(resultsR[:,2], relQ[:,1]))

                precisions = [] # List of precisions at each relevant&received document
                ind=1
                hits = 0
                for r in resultsAvg[:,2]:
                    if r in relQ[:,1]:
                        hits+=1
                        precisions.append(hits/ind)

                    ind+=1

                if len(precisions) == 0:
                    AvgP = 0
                else:
                    AvgP = sum(precisions)/len(precisions)

                # Discounted Cumulative Gain
                DCG20 = 0
                iDGC = 0
                for i, r in enumerate(results20[:,2]):
                    if i == 0:
                        iDGC = 1       
                        if len(np.where(relQ[:,1]==r)[0]) != 0:
                            DCG20 = relQ[np.where(relQ[:,1]==r)[0],2][0]
                        else:
                            DCG20 = 0
                    else:
                        # iDGC+=3/np.log2(i+1)
                        if len(np.where(relQ[:,1]==r)[0]) != 0:
                            DCG20 += relQ[np.where(relQ[:,1]==r)[0],2][0]/np.log2(i+1)
                            iDGC+= relQ[np.where(relQ[:,1]==r)[0],2][0]
                            print(relQ[np.where(relQ[:,1]==r)[0],2][0])
                            if i == 9:
                                if DCG20 == 0:
                                    nDCG10 = 0
                                else:
                                    nDCG10 = DCG20/(iDGC)
                                self.index[system+1][query+1]['nDCGa10'] = round(nDCG10, 3)
                        else:
                            DCG20 += 0

                if DCG20 == 0:
                    nDCG20 = 0
                else:
                    nDCG20 = DCG20/(iDGC)
                
                self.index[system+1][query+1]['Pa10'] = RaR10/10
                self.index[system+1][query+1]['Ra50'] = round(RaR50/len(relQ),3)
                self.index[system+1][query+1]['r-precision'] = round(RaRR/len(relQ),3)
                self.index[system+1][query+1]['AP'] = round(AvgP, 2)
                self.index[system+1][query+1]['nDCGa20'] = round(nDCG20, 3)

class text_eval():

    def __init__(self, path, stopwords):

        if stopwords is not None: # We may not give stopwords as an argument
            with open(stopwords,'r') as f: # Parse stopwords
                doc = f.read()
                Pattern = "\n" 
                lst = re.split(Pattern, doc)
                self.stopwords = [elem for elem in lst if elem != '']

        self.corpus_df = pd.read_csv(path,'\t', quoting=csv.QUOTE_NONE)
        line = []
        line.insert(0, {self.corpus_df.columns.values[0]: self.corpus_df.columns.values[0], self.corpus_df.columns.values[1]: self.corpus_df.columns.values[1]})
        # concatenate two dataframe
        self.corpus_df = pd.concat([pd.DataFrame(line),self.corpus_df], ignore_index=True)
        self.corpus_df = self.corpus_df.rename(columns={self.corpus_df.columns.values[0]:'Corpus', self.corpus_df.columns.values[1]:'Verse'})

    def pre_process(self, pstem = False): # Preprocess data

        ps = PorterStemmer()

        for row in range(self.corpus_df.shape[0]):

            self.corpus_df.iloc[row]['Verse'] = re.split('[^A-Z^a-z\d]', self.corpus_df.iloc[row]['Verse']) 
            self.corpus_df.iloc[row]['Verse'] = [elem.lower() for elem in self.corpus_df.iloc[row]['Verse'] if elem != ''] # Tokenisation
            self.corpus_df.iloc[row]['Verse'] = [elem for elem in self.corpus_df.iloc[row]['Verse'] if not(elem in self.stopwords)] # Remove stopwords
            if pstem == True:
                self.corpus_df.iloc[row]['Verse'] = [ps.stem(word).lower() for word in self.corpus_df.iloc[row]['Verse']] # Porter stemming

    def MICHI(self):
        self.tokens = []

        for x in self.corpus_df['Verse']:
            self.tokens += x

        self.tokens = list(set(self.tokens))
        self.michi = {} # Dictionary where each key is the token and contains a dict where keys are corpus and values are MI(index 0) CHI(index 1)
        
        for i, token in enumerate(self.tokens):
            self.michi[token] = {}
            for corpus in list(set(self.corpus_df['Corpus'])):
                    MI = 0
                    chi = 0
                    print(i)
                    self.michi[token][corpus] = {}
                    corp_docs = self.corpus_df[self.corpus_df['Corpus']==corpus]['Verse']
                    non_corp_docs = self.corpus_df[self.corpus_df['Corpus']!=corpus]['Verse']

                    N = self.corpus_df.shape[0] # Total number of documents. Stays constant.
                    N11 = len([1 for x in corp_docs if token in x])
                    N10 = len([1 for x in non_corp_docs if token in x])
                    N01 = len([1 for x in corp_docs if token not in x])
                    N00 = len([1 for x in non_corp_docs if token not in x])

                    try:
                        MI = (N11/N)*math.log((N*N11)/((N11+N10)*(N11+N01)),2)+ (N01/N)*math.log((N*N01)/((N01+N00)*(N11+N01)),2)+ (N10/N)*math.log((N*N10)/((N10+N11)*(N00+N10)),2)+ (N00/N)*math.log((N*N00)/((N10+N00)*(N01+N00)),2)
                    except ValueError:
                        self.michi[token][corpus]['MI'] = 0
                    try:
                        if N11 == 0:
                            chi = 0
                        else:
                            E00 = N*((N00+N10)/N)*((N00+N01)/N)
                            E01 = N*((N01+N11)/N)*((N01+N00)/N)
                            E10 = N*((N10+N11)/N)*((N10+N00)/N)
                            E11 = N*((N11+N10)/N)*((N11+N01)/N)

                            chi = ((N11-E11)**2)/E11 + ((N00-E00)**2)/E00 + ((N01-E01)**2)/E01 + ((N10-E10)**2)/E10 
                    except:
                        chi = 0

                    self.michi[token][corpus]['MI'] = MI
                    self.michi[token][corpus]['chi'] = chi

class text_classifier():

    def __init__(self, corpus):

        self.corpus = corpus

    def pre_process(self, pstem = False): # Preprocess data

        ps = PorterStemmer()

        for row in range(self.corpus_df.shape[0]):

            self.corpus_df.iloc[row]['Verse'] = re.split('[^A-Z^a-z\d]', self.corpus_df.iloc[row]['Verse']) 
            self.corpus_df.iloc[row]['Verse'] = [elem.lower() for elem in self.corpus_df.iloc[row]['Verse'] if elem != ''] # Tokenisation
            self.corpus_df.iloc[row]['Verse'] = [elem for elem in self.corpus_df.iloc[row]['Verse'] if not(elem in self.stopwords)] # Remove stopwords
            if pstem == True:
                self.corpus_df.iloc[row]['Verse'] = [ps.stem(word).lower() for word in self.corpus_df.iloc[row]['Verse']] # Porter stemming

    def get_vocab(self, corpus):

        vocab = []
        title = []

        for verse in corpus['Verse']:
            vocab+=verse

        for tit in corpus['Corpus']:
            title.append(tit)

        words = list(set(vocab))
        titles = list(set(title))


        return words, titles


    def word_cat_2ID(self,words, titles):
        # convert the vocab to a word id lookup dictionary
        # anything not in this will be considered "out of vocabulary" OOV
        word2id = {}
        for word_id,word in enumerate(words):
            word2id[word] = word_id
            
        # and do the same for the categories
        cat2id = {}
        for cat_id,cat in enumerate(titles):
            cat2id[cat] = cat_id

        self.word_id = word2id
        self.cat_id = cat2id

        return word2id, cat2id

    def shuffle_and_split(self,corpus):

        shuffled_corpus = corpus.sample(frac=1) # Shuffles the dataframe

        Xtrain, Xtest = train_test_split(shuffled_corpus, test_size=0.2)

        return Xtrain, Xtest

    def convert_to_bow(self,corpus,word2id):

        # matrix size is number of docs x vocab size + 1 (for OOV)
        matrix_size = (len(corpus['Verse']),len(word2id)+1)
        oov_index = len(word2id)
        # matrix indexed by [doc_id, token_id]
        X = scipy.sparse.dok_matrix(matrix_size)

        # iterate through all documents in the dataset
        for doc_id,doc in enumerate(corpus['Verse']):
            for word in doc:
                # default is 0, so just add to the count for this word in this doc
                # if the word is oov, increment the oov_index
                X[doc_id,word2id.get(word,oov_index)] += 1
        
        return X

    def catID(self,categories, cat2id):

        return [cat2id[x] for x in categories]

    def train_model(self, BOW_matrix, y_train, c):

        model = svm.LinearSVC(C=c)
        # then train the model!
        model.fit(BOW_matrix,y_train)

        return model

    def compute_accuracy(self, predictions, true_values):
        num_correct = 0
        num_total = len(predictions)
        for predicted,true in zip(predictions,true_values):
            if predicted==true:
                num_correct += 1
        return num_correct / num_total

    # def compute_stats(self, predictions, true_output:



def main():
    # evl = eval('system_results.csv', 'qrels.csv')
    # evl.create_index()
    # evl.stats()
    # hello = str(evl.index)
    # print(hello)

    testing = text_eval('test.tsv', stopwords='../../Collections/stopwords.txt')
    testing.pre_process()
    testing_corpus = testing.corpus_df.copy(deep=True)
    
    t_evl = text_eval("train_and_dev.tsv", stopwords='../../Collections/stopwords.txt')
    t_evl.pre_process()

    corpus_data = t_evl.corpus_df.copy(deep=True)

    print(corpus_data)

    text_class = text_classifier(corpus_data)

    print(text_class.corpus)

    vocab,titles = text_class.get_vocab(text_class.corpus)

    wordID, catID = text_class.word_cat_2ID(vocab,titles)

    Xtrain, Xtest = text_class.shuffle_and_split(text_class.corpus)

    BOW_matrix_train = text_class.convert_to_bow(Xtrain, wordID)

    BOW_matrix_test = text_class.convert_to_bow(Xtest, wordID)

    BOW_matrix_test2 = text_class.convert_to_bow(testing_corpus, wordID)

    print(catID)

    Ytrain = text_class.catID(Xtrain['Corpus'], catID)

    Ytest = text_class.catID(Xtest['Corpus'], catID)

    Ytest2 = text_class.catID(testing_corpus['Corpus'], catID)

    model = text_class.train_model(BOW_matrix_train,Ytrain, 3)

    predictions = model.predict(BOW_matrix_train)

    print(text_class.compute_accuracy(predictions,Ytrain))

    predictions = model.predict(BOW_matrix_test)

    print(text_class.compute_accuracy(predictions,Ytest))

    predictions = model.predict(BOW_matrix_test2)

    print(text_class.compute_accuracy(predictions,Ytest2))



if __name__ == '__main__':
    main()
