import pandas as pd
import numpy as np

class eval():

    def __init__(self, path, path2):

        self.system_df = pd.read_csv(path)
        self.qrels_df = pd.read_csv(path2)
        self.sys_np = self.system_df.to_numpy()
        self.q_np = self.qrels_df.to_numpy() 

    def create_index(self):
        self.index = {} # Each system will be a key

        for system in range(int(np.max(self.sys_np[:,0]))):
            self.index[system+1] = {} # Each query will be a key
            for query in range(int(np.max(self.sys_np[:,1]))):
                self.index[system+1][query+1] = {'Pa10':0,'Ra50':0,'r-precision':0,'AP':0,'nDCGa10':0,'nDCGa20':0} # Instance for each query per system

def main():
    evl = eval('system_results.csv', 'qrels.csv')
    evl.create_index()
    print(evl.index.keys())


if __name__ == '__main__':
    main()
