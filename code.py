from nltk.util import transitive_closure
from numpy import genfromtxt
import numpy as np
import re
from numpy.lib.function_base import average
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
from sklearn.metrics import confusion_matrix
import ast
import operator
from gensim.models import ldamodel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import datapath
import gensim.corpora as corpora


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
                    AvgP = sum(precisions)/relQ.shape[0]

                # Discounted Cumulative Gain
                DCG20 = []
                iDGC = []

                for i, r in enumerate(results20[:,2]):
                    if i == 0:
                        iDGC.append(relQ[i,2])
                        if len(np.where(relQ[:,1]==r)[0]) != 0:
                            DCG20.append(relQ[np.where(relQ[:,1]==r)[0],2][0])     
                        else:
                            DCG20.append(0)
                    else:
                        if len(np.where(relQ[:,1]==r)[0]) != 0:
                            DCG20.append(relQ[np.where(relQ[:,1]==r)[0],2][0]/np.log2(i+1))

                            if i+1 > relQ.shape[0]:
                                iDGC.append(0)
                            else:
                                iDGC.append(relQ[i,2]/np.log2(i+1))
                        else:
                            DCG20.append(0)
                            if i+1 > relQ.shape[0]:
                                iDGC.append(0)
                            else:
                                iDGC.append(relQ[i,2]/np.log2(i+1))

                    if i == 9: # We must assign iDCG10
                        nDCG10 = sum(DCG20)/sum(iDGC)
                        self.index[system+1][query+1]['nDCGa10'] = round(nDCG10, 3)

                    if i == 19:
                        if sum(DCG20) == 0:
                            nDCG20 = 0
                        else:
                            nDCG20 = sum(DCG20)/sum(iDGC)

                print(DCG20)
                print(iDGC)
                        
                    
                
                self.index[system+1][query+1]['Pa10'] = RaR10/10
                self.index[system+1][query+1]['Ra50'] = round(RaR50/len(relQ),3)
                self.index[system+1][query+1]['r-precision'] = round(RaRR/len(relQ),3)
                self.index[system+1][query+1]['AP'] = round(AvgP, 3)
                self.index[system+1][query+1]['nDCGa20'] = round(nDCG20, 3)

    def write_to_csv(self):
        results = "system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20" + "\n"

        for system in self.index:
            averages = {'system_number':str(system),'mean':'mean','Pa10':0,'Ra50':0,'r-precision':0,'AP':0,'nDCGa10':0,'nDCGa20':0}
            for query in self.index[system]:
                results += str(system) + "," + str(query) + "," + str(self.index[system][query]['Pa10']) + "," + str(self.index[system][query]['Ra50']) + "," + str(self.index[system][query]['r-precision']) + "," + str(self.index[system][query]['AP']) + "," + str(self.index[system][query]['nDCGa10']) + ',' + str(self.index[system][query]['nDCGa20']) + '\n'
                averages['Pa10']+= self.index[system][query]['Pa10']
                averages['Ra50']+= self.index[system][query]['Ra50']
                averages['r-precision']+= self.index[system][query]['r-precision']
                averages['AP']+= self.index[system][query]['AP']
                averages['nDCGa10']+= self.index[system][query]['nDCGa10']
                averages['nDCGa20']+= self.index[system][query]['nDCGa20']

            results += str(system) + "," + 'mean' + "," + str(averages['Pa10']/10) + "," + str(averages['Ra50']/10) + "," + str(averages['r-precision']/10) + "," + str(averages['AP']/10) + "," + str(averages['nDCGa10']/10) + ',' + str(averages['nDCGa20']/10) + '\n'

        with open('ir_eval.csv','w') as f:
            f.write(results)
            f.close()


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

        nvals = ""

        for x in self.corpus_df['Verse']:
            self.tokens += x

        self.tokens = list(set(self.tokens))
        self.corpussies = list(set(self.corpus_df['Corpus']))
        self.michi = {} # Dictionary where each key is the token and contains a dict where keys are corpus and values are MI(index 0) CHI(index 1)
        
        print(len(self.tokens))

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

                nvals += str(token) + ", " + str(corpus) +  ", "  + "N: " + str(N) + ", " + "N11: " + str(N11) + ", " + "N10: " + str(N10) + ", " + "N01: " + str(N01) + ", " + "N00: " + str(N00) + "\n"


                try:
                    one = (N11/N)*math.log((N*N11)/((N11+N10)*(N11+N01)),2)
                except:
                    one = 0
                try:
                    two = (N01/N)*math.log((N*N01)/((N01+N00)*(N11+N01)),2)
                except:
                    two = 0
                try:
                    three = (N10/N)*math.log((N*N10)/((N10+N11)*(N00+N10)),2)
                except:
                    three = 0
                try:
                    four = (N00/N)*math.log((N*N00)/((N10+N00)*(N01+N00)),2)
                except:
                    four = 0

                MI = one+two+three+four

                E00 = N*((N00+N10)/N)*((N00+N01)/N)
                E01 = N*((N01+N11)/N)*((N01+N00)/N)
                E10 = N*((N10+N11)/N)*((N10+N00)/N)
                E11 = N*((N11+N10)/N)*((N11+N01)/N)
                
                # try:
                #     one = ((N11-E11)**2)/E11
                # except:
                #     one = 0
                # try:
                #     two = ((N01-E01)**2)/E01
                # except:
                #     two = 0
                # try:
                #     three = ((N10-E10)**2)/E10
                # except:
                #     three = 0

                chi =  ((N11-E11)**2)/E11 + ((N01-E01)**2)/E01 + ((N10-E10)**2)/E10
                

                self.michi[token][corpus]['MI'] = round(MI,3)
                self.michi[token][corpus]['chi'] = round(chi,3)


    def michi_to_csv(self, michi):
        
        # with open(file, 'r') as f:
        #     string = f.read()
        #     f.close()
        
        michi = michi

        MIOT = {}
        MINT = {}
        MIQR = {}

        mi_dict = {'NT':MINT, 'OT':MIOT, 'Quran':MIQR}

        CHIOT = {}
        CHINT = {}
        CHIQR = {}

        chi_dict = {'NT':CHINT, 'OT':CHIOT, 'Quran':CHIQR}

        for token in michi:
            for book in michi[token]:
                mi_dict[book][token] = michi[token][book]['MI']
                chi_dict[book][token] = michi[token][book]['chi']

        MIOT = text_eval.sort_dict(MIOT)
        MINT = text_eval.sort_dict(MINT)
        MIQR = text_eval.sort_dict(MIQR)

        CHIQR = text_eval.sort_dict(CHIQR)
        CHINT = text_eval.sort_dict(CHINT)
        CHIOT = text_eval.sort_dict(CHIOT)

        text_eval.write_p2_file(MIOT, 'MIOT')
        text_eval.write_p2_file(MINT, 'MINT')
        text_eval.write_p2_file(MIQR, 'MIQR')
        text_eval.write_p2_file(CHIQR, 'CHIQR')
        text_eval.write_p2_file(CHINT, 'CHINT')
        text_eval.write_p2_file(CHIOT, 'CHIOT')

        return MIOT, MINT, MIQR, CHIOT, CHINT, CHIQR
        
    def write_p2_file(dict, filename):

        writing_str = ""

        with open(filename+'.txt','w') as f:
            for token in dict.keys():
                writing_str+= str(token) + "," + str(dict[token]) + '\n'

            f.write(writing_str)

    def sort_dict(dicky):

        dicky = dict(sorted(dicky.items(), key=operator.itemgetter(1), reverse=True))

        return dicky
        

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

    def ext_pre_process(corpus, topMI):

        ps = PorterStemmer()

        corpus2= corpus.copy(deep=True)

        for row in range(corpus.shape[0]):
            corpus2.iloc[row]['Verse'] = [elem.lower() for elem in corpus2.iloc[row]['Verse'] if elem != ''] # Tokenisation
            corpus2.iloc[row]['Verse'] = [elem for elem in corpus2.iloc[row]['Verse'] if elem in topMI] # Remove stopwords
            corpus2.iloc[row]['Verse'] = [ps.stem(word).lower() for word in corpus2.iloc[row]['Verse']] # Porter stemming

        return corpus2

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
        id2cat = {}
        for cat_id,cat in enumerate(titles):
            cat2id[cat] = cat_id
            id2cat[cat_id] = cat

        self.word_id = word2id
        self.cat_id = cat2id
        self.id2cat = id2cat

        return word2id, cat2id

    def shuffle_and_split(self,corpus, split):

        shuffled_corpus = corpus.sample(frac=1) # Shuffles the dataframe

        Xtrain, Xtest = train_test_split(shuffled_corpus, test_size=split)

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

    def train_model(self, BOW_matrix, y_train, c, lin = True):

        if lin == False:
            model = svm.SVC(C=c, kernel='rbf')
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

    def compute_stats(self, predictions, true_output):

        cm = confusion_matrix(predictions, true_output)

        rates = {'TP':{},'FP':{},'TN':{},'FN':{}}

        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        Precision = TP/(TP + FP)
        Recall = TP/(TP + FN)
        f1 = 2*((Precision*Recall)/(Precision+Recall))

        Precision = np.round(np.append(Precision, np.average(Precision)),3)
        Recall = np.round(np.append(Recall, np.average(Recall)),3)
        f1 = np.round(np.append(f1, np.average(f1)),3)

        
        return Precision, Recall, f1


def main():

    # ############# TASK 1 #############################

    evl = eval('system_results.csv', 'qrels.csv')
    evl.create_index()
    evl.stats()
    evl.write_to_csv()

    # ################ END #################################

    ############### TASK2 ##############################
    
    t_evl = text_eval("train_and_dev.tsv", stopwords='../../Collections/stopwords.txt')
    t_evl.pre_process(pstem=True)
    t_evl.MICHI()
    t_evl.michi_to_csv(t_evl.michi)

    ############### END. NOTE LDA AT BOTTOM ##############################

    # Pre-process the development (unseen) data
    dev = text_eval('train_and_dev.tsv',stopwords='../../Collections/stopwords.txt')

    dev.pre_process()

    #  Make hard copy of previous corpus
    corpus_data = t_evl.corpus_df.copy(deep=True)

    #### RUN THIS FOR PART 3 ##########  corpus_data = the pre_processed data from part 1

    # Create instance to use the data processing methods
    text_class = text_classifier(corpus_data)

    # Convert the text to tokens and get classes
    vocab,titles = text_class.get_vocab(text_class.corpus)

    # Create unique work to ID and category to ID mapping
    wordID, catID = text_class.word_cat_2ID(vocab,titles)

    # Split into train and test.
    Xtrain, Xtest = text_class.shuffle_and_split(text_class.corpus, 0.20)
    Xitrain, Xitest = text_class.shuffle_and_split(text_class.corpus, 0.10)

    # Testing, Training and Development sets converted to bag of words
    BOW_matrix_train = text_class.convert_to_bow(Xtrain, wordID)
    BOW_matrix_test = text_class.convert_to_bow(Xtest, wordID)
    BOW_matrix_dev = text_class.convert_to_bow(dev.corpus_df, wordID)

    # IMPROVED MODEL 
    BOW_matrix_itrain = text_class.convert_to_bow(Xitrain, wordID)
    BOW_matrix_itest = text_class.convert_to_bow(Xitest, wordID)

    # Convert the corpus titles to a numerical value. 
    Ytrain = text_class.catID(Xtrain['Corpus'], catID)
    Ytest = text_class.catID(Xtest['Corpus'], catID)
    Ydev = text_class.catID(dev.corpus_df['Corpus'], catID)

    # Get improved Y vectors
    Yitrain = text_class.catID(Xitrain['Corpus'], catID)
    Yitest = text_class.catID(Xitest['Corpus'], catID)

    # Strain the svm
    model = text_class.train_model(BOW_matrix_train,Ytrain, 3)
    i_model = text_class.train_model(BOW_matrix_itrain,Yitrain, 3, lin=False)

    # Run the svm on training, testing and development sets.
    predictions1 = model.predict(BOW_matrix_train)
    predictions2 = model.predict(BOW_matrix_test)
    predictions3 = model.predict(BOW_matrix_dev)

    # Run the improved svm on training, testing and development sets.
    predictionsi1 = i_model.predict(BOW_matrix_itrain)
    predictionsi2 = i_model.predict(BOW_matrix_itest)
    predictionsi3 = i_model.predict(BOW_matrix_dev)

    # # Obtain accuracy among other stats
    print('Improved')

    print(text_class.compute_accuracy(predictionsi1,Yitrain))
    print(text_class.compute_accuracy(predictionsi2,Yitest))
    print(text_class.compute_accuracy(predictionsi3, Ydev))

    print('Baseline:')

    print(text_class.compute_accuracy(predictions1,Ytrain))
    print(text_class.compute_accuracy(predictions2,Ytest))
    print(text_class.compute_accuracy(predictions3, Ydev))

    Precisiontr, Recalltr, f1tr = text_class.compute_stats(predictions1, Ytrain)
    PrecisionTst, RecallTst, f1Tst = text_class.compute_stats(predictions2, Ytest)
    Precisiond, Recalld, f1d = text_class.compute_stats(predictions3, Ydev)

    Precisiontri, Recalltri, f1tri = text_class.compute_stats(predictionsi1, Yitrain)
    PrecisionTsti, RecallTsti, f1Tsti = text_class.compute_stats(predictionsi2, Yitest)
    Precisiondi, Recalldi, f1di = text_class.compute_stats(predictionsi3, Ydev)

    # Building the file for task 3
    task3 = "system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro" + "\n"
    models = ['baseline,train,', 'baseline,dev,','baseline,test,', 'improved,train,', 'improved,dev,', 'improved,test,'] # Add improved models
    
    for i,model in enumerate(models):
        task3+=model

        a = catID['Quran']
        j = catID['OT']
        k = catID['NT']

        if i == 0:
            task3+= str(Precisiontr[a]) + "," + str(Recalltr[a]) + "," + str(f1tr[a]) + "," + str(Precisiontr[j]) + "," + str(Recalltr[j]) + "," + str(f1tr[j]) + "," + str(Precisiontr[k]) + "," + str(Recalltr[k]) + "," + str(f1tr[k]) + "," + str(Precisiontr[3]) + "," + str(Recalltr[3]) + "," + str(f1tr[3]) + "\n"

        if i == 1:
            task3+= str(Precisiond[a]) + "," + str(Recalld[a]) + "," + str(f1d[a]) + "," + str(Precisiond[j]) + "," + str(Recalld[j]) + "," + str(f1d[j]) + "," + str(Precisiond[k]) + "," + str(Recalld[k]) + "," + str(f1d[k]) + "," + str(Precisiond[3]) + "," + str(Recalld[3]) + "," + str(f1d[3]) + "\n"

        if i == 2:
            task3+= str(PrecisionTst[a]) + "," + str(RecallTst[a]) + "," + str(f1Tst[a]) + "," + str(PrecisionTst[j]) + "," + str(RecallTst[j]) + "," + str(f1Tst[j]) + "," + str(PrecisionTst[k]) + "," + str(RecallTst[k]) + "," + str(f1Tst[k]) + "," + str(PrecisionTst[3]) + "," + str(RecallTst[3]) + "," + str(f1Tst[3]) + "\n"

        if i == 3:
            task3+= str(Precisiontri[a]) + "," + str(Recalltri[a]) + "," + str(f1tri[a]) + "," + str(Precisiontri[j]) + "," + str(Recalltri[j]) + "," + str(f1tri[j]) + "," + str(Precisiontri[k]) + "," + str(Recalltri[k]) + "," + str(f1tri[k]) + "," + str(Precisiontri[3]) + "," + str(Recalltri[3]) + "," + str(f1tri[3]) + "\n"

        if i == 4:
            task3+= str(PrecisionTsti[a]) + "," + str(RecallTsti[a]) + "," + str(f1Tsti[a]) + "," + str(PrecisionTsti[j]) + "," + str(RecallTsti[j]) + "," + str(f1Tsti[j]) + "," + str(PrecisionTsti[k]) + "," + str(RecallTsti[k]) + "," + str(f1Tsti[k]) + "," + str(PrecisionTsti[3]) + "," + str(RecallTsti[3]) + "," + str(f1Tsti[3]) + "\n"

        if i == 5:
            task3+= str(Precisiondi[a]) + "," + str(Recalldi[a]) + "," + str(f1di[a]) + "," + str(Precisiondi[j]) + "," + str(Recalldi[j]) + "," + str(f1di[j]) + "," + str(Precisiondi[k]) + "," + str(Recalldi[k]) + "," + str(f1di[k]) + "," + str(Precisiondi[3]) + "," + str(Recalldi[3]) + "," + str(f1di[3]) + "\n"

    with open('classification.csv','w') as f:
        f.write(task3)
        f.close()

    ####### PART 3 END ##########

    # Create a corpus from a list of texts

    #
    #
    # LDA BEGIN!!!
    #
    #

    texts = [list(x) for x in t_evl.corpus_df['Verse'].values]

    id2word = corpora.Dictionary(texts)
    
    corpus = []
    for text in texts:
        new = id2word.doc2bow(text)
        corpus.append(new)

    lda_model = ldamodel.LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=20,
                                update_every=1,
                                chunksize=100,
                                passes=10,
                                alpha='auto',
                                )

    docs1 = [list(x) for x in t_evl.corpus_df[t_evl.corpus_df['Corpus']=='Quran']['Verse'].values]
    docs2 = [list(x) for x in t_evl.corpus_df[t_evl.corpus_df['Corpus']=='OT']['Verse'].values]
    docs3 = [list(x) for x in t_evl.corpus_df[t_evl.corpus_df['Corpus']=='NT']['Verse'].values]

    avg_score = {}

    for x in range(lda_model.num_topics):
        avg_score[x]=0

    for doc in docs1:
        instance = id2word.doc2bow(doc)
        ps = lda_model[instance]

        for t,p in ps:
            avg_score[t]+=p

    for t in avg_score.keys():
        avg_score[t] = avg_score[t]/len(docs1)

    avg_score2 = {}

    for x in range(lda_model.num_topics):
        avg_score2[x]=0

    for doc in docs2:
        instance = id2word.doc2bow(doc)
        ps = lda_model[instance]

        for t,p in ps:
            avg_score2[t]+=p

    for t in avg_score2.keys():
        avg_score2[t] = avg_score2[t]/len(docs2)

    avg_score3 = {}

    for x in range(lda_model.num_topics):
        avg_score3[x]=0

    for doc in docs3:
        instance = id2word.doc2bow(doc)
        ps = lda_model[instance]

        for t,p in ps:
            avg_score3[t]+=p

    for t in avg_score3.keys():
        avg_score3[t] = avg_score3[t]/len(docs3)

    quran_topic = max(avg_score, key=avg_score.get)
    OT_topic = max(avg_score2, key=avg_score2.get)
    NT_topic = max(avg_score3, key=avg_score3.get)


    string = ""
    for idx, topic in lda_model.print_topics(-1):

        parsed = [str(x) for x in re.split('[^A-Z^a-z\d]', topic)]

        topic = ""
        for p in parsed:
            topic += p + " "

        if idx == quran_topic:
            string+= "Quran most likely topic and terms: " + topic + "\n"
        if idx == OT_topic:
            string+= "OT most likely topic and terms: " + topic + "\n"
        if idx == NT_topic:
            string+= "NT most likely topic and terms: " + topic

    with open('LDA_results.txt', 'w') as f:
        f.write(string)
        f.close()
    
    
    
    # LDA END !!!!!!!!
    
    

if __name__ == '__main__':
    main()
