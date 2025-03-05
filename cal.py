import numpy as np
from scipy import linalg
import random
import os
import matplotlib.pyplot as plt

def betagenerator(p,k,beta01,beta02):
    np.random.seed(0)
    true = np.random.choice(np.arange(p),size=k,replace=False)
    np.random.seed()
    beta0 = np.zeros(p)
    if k % 2 == 0:
        true_beta = np.repeat([beta01,beta02],k//2)
    else:
        true_beta = np.append(np.tile(np.array([beta01,beta02]), (k-1)//2),beta01)
        
    beta0[true] = true_beta
    return beta0

def betagenerator_aligned(p,k,beta01,beta02):
    
    true = np.arange(k)
    beta0 = np.zeros(p)
    if k % 2 == 0:
        true_beta = np.repeat([beta01,beta02],k//2)
        #true_beta = np.tile(np.array([beta01,beta02]), k//2) #1,-1,1,-1
    else:
        true_beta = np.append(np.tile(np.array([beta01,beta02]), (k-1)//2),beta01)
        
    beta0[true] = true_beta
    return beta0

def generate_toeplitz(p):
    p=int(p)
    # Define the blockwise diagonal matrix of 10 identical unit diagonal Toeplitz matrices
    block = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            block[i, j] = 6 * (p - abs(i-j)) / (10 * (p - 1))
    np.fill_diagonal(block, 1)

    # Construct the covariance matrix
    covariance_matrix = np.zeros((p * 10, p * 10))
    for i in range(10):
        covariance_matrix[i * p:(i + 1) * p, i * p:(i + 1) * p] = block

    return covariance_matrix

def cholesky_root(corst, p, rho):
    if corst == "ind":
        Root = np.diag(np.ones(p))
    if corst == "toeplitz":
        Root = linalg.cholesky(generate_toeplitz(p/10)).T
    elif corst == "AR-1":
        Root = np.zeros((p, p))
        Root[0, :] = rho ** np.arange(p)
        c = np.sqrt(1 - rho ** 2)
        R2 = c * Root[0, :]
        for j in range(1, p):
            Root[j, j:] = R2[:p-j]
        Root = Root.T
    return Root

def datagenerator(NB, p,B,corst_x, rho_x, beta0, family,factor,seed):
    np.random.seed(seed)
    X_total = (cholesky_root(corst=corst_x,p=p,rho=rho_x) @ np.random.normal(0,1,size=(p,B*NB))).T 
    
    if family == "binomial":
        X_total = X_total *factor
        y = np.random.binomial(1, sigmoid(X_total @ beta0))

    elif family == "gaussian":
        epsilon = np.random.normal(0,scale=1,size=B*NB)
        y = X_total @ beta0 + epsilon

    elif family == "poisson":
        X_total = X_total * factor
        eta = X_total @ beta0 
        lambda_ = np.exp(eta)
        y = np.random.poisson(lambda_)

    X = np.reshape(X_total, (NB,B,p))
    y = np.reshape(y, (NB,B,1))

    return X,X_total, y

def mean_est(X, beta, family="gaussian"):
    lp = X @ beta
    if family == "gaussian":
        average = lp

    elif family == "binomial":
        average = np.exp(lp) / (1 + np.exp(lp))

    elif family == "poisson":
        average = np.exp(lp)
    else:
        raise ValueError("Unknown distribution family")
    return average

def var_est(X, beta, family):
    lp = X @ beta
    if family == "gaussian":
        vu = np.ones(X.shape[0])

    elif family == "binomial":
        vu = np.exp(lp) / (1 + np.exp(lp)) ** 2

    elif family == "poisson":
        vu = np.exp(lp)

    else:
        raise ValueError("Unknown distribution family")
    return vu


def phi(lp, family):
    if family == "gaussian":
        return 0.5 * lp**2

    elif family == "binomial":
        return np.log1p(np.exp(lp))

    elif family == "poisson":
        return np.exp(lp)

    else:
        raise ValueError("Unknown family")

def negative_log_likelihood(y, X, beta, family):

    n = X.shape[0]
    lp = X @ beta
    
    if family == "gaussian":
        return (y-lp)**2/ n

    elif family == "binomial":
        term1 = y * lp
        term2 = phi(lp, family)
        nll = -np.sum(term1 - term2) / n
        return nll

    elif family == "poisson":
        term1 = y * lp
        term2 = phi(lp, family)
        nll = -np.sum(term1 - term2) / n
        return nll
    else:
        raise ValueError("Unknown family")

def regularizer(x,lam,penalty):
    if penalty == "l1":
        return lam*np.sign(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#gives summary of one single simulation
def summary(debeta_trajec,lb_trajec,ub_trajec,tao_trajec,out_index,beta0,subset,beta01,beta02):

    beta_real = beta0[subset]
    A1 = np.where((beta_real == 0))[0].tolist()  #groupping by true beta
    A2 = np.where((beta_real == beta01))[0].tolist()
    A3 = np.where((beta_real == beta02))[0].tolist()

    BIAS = []
    TAO = []
    LENGTH = []
    CP = []

    for o in out_index:

        debeta = debeta_trajec[:,o]
        lb = lb_trajec[:,o]
        ub = ub_trajec[:,o]
        tao = tao_trajec[:,o]

        cp = []
        bias = []
        length = []
        ta = []

        for A in [A1,A2,A3]:

            count = 0
            total_bias = 0
            total_length = 0
            total_tao = 0
            for i in range(len(A)):
                r= A[i]
        
                if lb[r] <= beta_real[r] <= ub[r]:
                    count += 1

                total_bias+= np.abs(beta_real[r] - debeta[r])
                total_length += (ub[r] - lb[r])
                total_tao += np.abs(tao[r])

            if len(A) > 0:
                cp.append(count/len(A))
                bias.append(total_bias/len(A))
                length.append(total_length/len(A))
                ta.append(total_tao/len(A))
    
        CP.append(np.array(cp))
        BIAS.append(np.array(bias))
        LENGTH.append(np.array(length))
        TAO.append(np.array(ta))

    return CP,BIAS,LENGTH,TAO

def plot1(DEBETA,LB,UB,NB,p,folder_name):
    xx=np.arange(NB)
    fig, axes = plt.subplots(1,3,figsize=(18, 5))

    ax=axes[0]
    ax.plot(xx,np.mean(DEBETA,axis=0)[0,:].T,color='C0')
    ax.fill_between(xx, np.mean(LB,axis=0)[0,:].T, np.mean(UB,axis=0)[0,:].T, alpha=0.2,color='C0')
    ax.set_xlim (0,NB)
    ax.set_ylim (-3,3)
    ax.grid()
    ax.set_title(r'$\beta^*_k = 0$',fontsize = 14)
    ax.set_ylabel(r'95\% confidence interval',fontsize = 14) 

    ax=axes[1]
    ax.plot(xx,np.mean(DEBETA,axis=0)[3,:].T,color='C1')
    ax.fill_between(xx, np.mean(LB,axis=0)[3,:].T, np.mean(UB,axis=0)[3,:].T, alpha=0.2,color='C1')
    ax.set_xlim (0,NB)
    ax.set_ylim (-3,3)
    ax.grid()
    ax.set_title(r'$\beta^*_k = 1$',fontsize = 14)

    ax=axes[2]
    ax.plot(xx,np.mean(DEBETA,axis=0)[6,:].T,color='C2')
    ax.fill_between(xx, np.mean(LB,axis=0)[6,:].T, np.mean(UB,axis=0)[6,:].T, alpha=0.2,color='C2')
    ax.set_xlim (0,NB)
    ax.set_ylim (-3,3)
    ax.grid()
    ax.set_title(r'$\beta^*_k = -1$',fontsize = 14)

    fig.text(0.52, 0.01, 'Sample size', ha='center',fontsize = 14)
    plt.savefig(os.path.join(folder_name, "result_p%s_n%s.png" % (p, NB)) )