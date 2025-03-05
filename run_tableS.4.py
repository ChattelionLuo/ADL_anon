import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os
import pandas as pd

import cal
import adl

n_sim = 500   #number of simulations

# GLM objective
family = "binomial" #"gaussian"
penalty = "l1"

# for data generation
corst_x = "toeplitz"    #"ind", "AR-1", "toeplitz"
rho_x = 0.6     #AR-1 correlation 

factor=1
beta01 = 0.5
beta02 = -0.5      #sparsity

folder_name = "tableS.4"
os.mkdir(folder_name)

LB= []
UB=[]
DEBETA=[]
TAO=[]
Summary= []
Ti=[]
    
for u in range(n_sim):
        
    beta0=cal.betagenerator_aligned(500,6,beta01,beta02)
    X,X_total,y=cal.datagenerator(200, 500, 1, corst_x, rho_x, beta0,family,np.sqrt(factor),u)

    random.seed(u)
    a0 = random.sample(np.where((beta0 == 0))[0].tolist(), 3)
    a1 = random.sample(np.where((beta0 == beta01))[0].tolist(), 3)
    a2 = random.sample(np.where((beta0 == beta02))[0].tolist(), 3)
    random.seed()
    subset=a0+a1+a2
        
    model=adl.on_ADL(subset,family,beta0,penalty)
    start_time = time.time()
    model.fit( X,y )
    Ti.append((time.time() - start_time))
    print("---%s-th simulation, %s seconds ---" % (u+1 ,(time.time() - start_time)) )

    report=[79,119,159,199]   #record results in the 80, 140, 200-th step
    summary=cal.summary(model.debeta_trajec,model.lb_trajec,model.ub_trajec,model.tao_trajec,np.arange(200),beta0,subset,beta01,beta02)
    Summary.append(summary)
    DEBETA.append(model.debeta_trajec)
    LB.append(model.lb_trajec)
    UB.append(model.ub_trajec)
    TAO.append(model.tao_trajec)

np.save(os.path.join(folder_name, 'lb.npy'), LB)
np.save(os.path.join(folder_name, 'ub.npy'), UB) 
np.save(os.path.join(folder_name, 'debeta.npy'), DEBETA)
np.save(os.path.join(folder_name, 'tao.npy'), TAO) 

column_names = ['beta*=0', 'beta*=1', 'beta*=-1']
row_names = ['80', '120', '160', '200']
block_names = ['Coverage Probability', 'Bias', 'Length', 'STD']

output = np.mean(np.array(Summary),axis=0)[:,report]
formatted_output = np.array([f'{x:.3f}' for x in output.flatten()]).reshape(output.shape)
with open(os.path.join(folder_name, "summary.txt"), "w") as f:
    for i, block in enumerate(block_names):
        f.write(f"{block}\n")
        block_data = formatted_output[i, :, :]
        block_df = pd.DataFrame(block_data, columns=column_names, index=row_names)
        block_df.to_csv(f, sep='\t', mode='a')
        f.write("\n")
    f.write(f"Mean Time per Simulation: {np.mean(Ti)}\n")


    
cal.plot1(DEBETA,LB,UB,200,500,folder_name)

print("finished")