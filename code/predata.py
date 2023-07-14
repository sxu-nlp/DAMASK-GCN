
import pandas as pd
from os.path import join

#Pre-process the data
path = '../data/ml-1m/'

trainData = pd.read_table(join(path, 'train.txt'), header=None,sep=' ').astype(int)
testData = pd.read_table(join(path, 'test.txt'), header=None,sep=' ').astype(int)
trainData -= 1
testData -= 1
trainData = trainData.astype(int)

trainData = trainData.astype(str)
trainData.to_csv(path+'model1/graph.txt', sep=' ', index=False,header=None)
testData = testData.astype(int)
testData = testData.astype(str)
testData.to_csv(path+'/test.txt', sep=' ', index=False,header=None)

