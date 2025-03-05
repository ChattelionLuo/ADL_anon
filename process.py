import os
import numpy as np
import pandas as pd

from itertools import tee, islice
from collections import Counter

from scipy.sparse import csc_array,hstack,vstack
from scipy import sparse

data=pd.read_csv("combined_data.csv")

def ngrams(lst, n):
  tlst = lst
  while True:
    a, b = tee(tlst)
    l = tuple(islice(a, n))
    if len(l) == n:
      yield l
      next(b)
      tlst = b
    else:
      break
    
os.mkdir("bigram_dataset")
num_dictionary=[]
for i in range(83440):  
        
        if data['label'][i] == 1:
             y=1
        else:
             y=0
        
        words = data['text'][i].split()#re.findall("\w+", data)
        word_frequency_i = Counter(ngrams(words, 1)) + Counter(ngrams(words, 2))
        if i == 0:
            cumulative_word_frequency = word_frequency_i
            word_frequency = cumulative_word_frequency
        else:
            word_frequency = cumulative_word_frequency + word_frequency_i
            word_frequency.subtract(cumulative_word_frequency)
            
            cumulative_word_frequency = cumulative_word_frequency + word_frequency_i
            
        num_dictionary.append(len(cumulative_word_frequency))    
            
        X = np.log(np.array(list(word_frequency.values())) + 1)
        total_data = np.insert(X, 0, y)
        total_data_pd = pd.DataFrame(total_data)
        total_data_pd.to_csv('bigram_dataset/data_%s.csv'%(i), index=False, header=False)
        print('data %s loaded'%(i))

print('extracted %s unique words'%(len(cumulative_word_frequency)))
mm= cumulative_word_frequency.keys()

#transform the data into a sparse matrix
for i in range(83439):
        xa= np.array(pd.read_csv('bigram_dataset/data_%s.csv'%(i), header=None))
        y=xa[0].reshape(1,1)
        XX = np.zeros((3644040, 1))
        XX[:int(xa[1:].shape[0])] = xa[1:]
        # Reshape the array to (1, 101)
        X = XX.reshape(1, -1)
        X = csc_array(X)
        if i == 0:
            X_total = X
            y_total = y
        else:
            X_total = vstack((X_total, X))
            y_total = np.concatenate((y_total, y), axis=0)
        print(i)
sparse.save_npz('bigram_X.npz', X_total.tocsc())
np.save('trec_bigram_y.npy',y_total)