import numpy as np
import multiprocessing
import os

from scipy import sparse
import adl_realdata

family = "binomial"
penalty = "l1"
X_total=sparse.load_npz('bigram_X.npz')
y_total=np.load('bigram_y.npy')

feature=[6795] # [7856], [22608] (user can choose the feature index from the following list)

'''
index for bigram features:
('investment',) 6795
('schedule',) 7856
('per', 'cent') 22608
'''

print(f'feature {feature[0]} started')
    
model=adl_realdata.on_ADL(feature,family,penalty)
folder_name = "./realdata_result/feature%s" % (model.subset[0])
os.makedirs(folder_name) # create the folder in the current directory

model.fit(X_total,y_total,folder_name)

np.save(os.path.join(folder_name, 'lb%s.npy'%model.subset[0]), model.lb_trajec)
np.save(os.path.join(folder_name, 'ub%s.npy'%model.subset[0]), model.ub_trajec)
np.save(os.path.join(folder_name, 'debeta%s.npy'%model.subset[0]), model.debeta_trajec)
np.save(os.path.join(folder_name, 'tao%s.npy'%model.subset[0]), model.tao_trajec)
np.save(os.path.join(folder_name, 'beta%s.npy'%model.subset[0]), model.beta_trajec)
np.save(os.path.join(folder_name, 'pred.npy'), np.array(model.pred_err))

print(f'feature {model.subset[0]} finised')
