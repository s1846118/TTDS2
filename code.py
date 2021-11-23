from numpy import genfromtxt
import numpy as np
import re
import pandas as pd
from nltk import text
from nltk.stem import PorterStemmer

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
        pass


def main():
    # evl = eval('system_results.csv', 'qrels.csv')
    # evl.create_index()
    # evl.stats()
    # print(evl.index)

    t_evl = text_eval("train_and_dev.tsv", stopwords='../../Collections/stopwords.txt')
    t_evl.pre_process()
    print(t_evl.corpus_df)



if __name__ == '__main__':
    main()
