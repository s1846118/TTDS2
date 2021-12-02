from numpy import genfromtxt
import numpy as np
import re
import pandas as pd
from nltk import text
from nltk.stem import PorterStemmer
import math

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

        self.corpus_df = pd.read_csv(path,'\t')
        line = []
        line.insert(0, {self.corpus_df.columns.values[0]: self.corpus_df.columns.values[0], self.corpus_df.columns.values[1]: self.corpus_df.columns.values[1]})
        # concatenate two dataframe
        self.corpus_df = pd.concat([pd.DataFrame(line),self.corpus_df], ignore_index=True)
        self.corpus_df = self.corpus_df.rename(columns={self.corpus_df.columns.values[0]:'Corpus', self.corpus_df.columns.values[1]:'Verse'})

    def pre_process(self): # Preprocess data

        ps = PorterStemmer()

        for row in range(self.corpus_df.shape[0]):

            self.corpus_df.iloc[row]['Verse'] = re.split('[^A-Z^a-z\d]', self.corpus_df.iloc[row]['Verse']) 
            self.corpus_df.iloc[row]['Verse'] = [elem.lower() for elem in self.corpus_df.iloc[row]['Verse'] if elem != ''] # Tokenisation
            self.corpus_df.iloc[row]['Verse'] = [elem for elem in self.corpus_df.iloc[row]['Verse'] if not(elem in self.stopwords)] # Remove stopwords
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

                    N, N11, N10, N01, N00 = 1, 1, 1, 1, 1

                    N = len(self.corpus_df['Corpus']) # Total number of documents. Stays constant.
                    N11 = len([1 for x in corp_docs if token in x])
                    N10 = len([1 for x in non_corp_docs if token in x])
                    N01 = len([1 for x in corp_docs if token not in x])
                    N00 = len([1 for x in non_corp_docs if token not in x])

                    try:
                        MI = (N11/N)*math.log((N*N11)/((N11+N10)*(N11+N01)),2)
                        + (N01/N)*math.log((N*N01)/((N01+N00)*(N11+N01)),2)
                        + (N10/N)*math.log((N*N10)/((N10+N11)*(N00+N10)),2)
                        + (N00/N)*math.log((N*N00)/((N10+N00)*(N01+N00)),2)
                    except ValueError:
                        self.michi[token][corpus]['MI'] = 0
                    try:
                        E00 = N*((N00+N10)/N)*((N00+N01)/N)
                        E01 = N*((N01+N11)/N)*((N01+N00)/N)
                        E10 = N*((N10+N11)/N)*((N10+N00)/N)
                        E11 = N*((N11+N10)/N)*((N11+N01)/N)

                        chi = ((N11-E11)**2)/E11 + ((N00-E00)**2)/E00 + ((N01-E01)**2)/E01 + ((N10-E10)**2)/E10 
                    except:
                        chi = 0

                    self.michi[token][corpus]['MI'] = MI
                    self.michi[token][corpus]['chi'] = chi

    # N11 (number of documents that contain the term and are in corpus 1)
    # N10 (number of documents that contain the term and are in corpus 2).
    # N01 (number of documents that do not contain the term and are in corpus 1)
    # N00 (number of documents that do not contain the term and are in corpus 2)


def main():
    # evl = eval('system_results.csv', 'qrels.csv')
    # evl.create_index()
    # evl.stats()
    # hello = str(evl.index)
    # print(hello)
    
    t_evl = text_eval("train_and_dev.tsv", stopwords='../../Collections/stopwords.txt')
    t_evl.pre_process()
    t_evl.MICHI()
    print(t_evl.michi)

    with open('michi.txt','w') as f:
        f.write(str(t_evl.michi))
        f.close()





if __name__ == '__main__':
    main()
