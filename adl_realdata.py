import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import cal
import radar
import random
import time
import os
import pandas as pd

import scipy
from scipy.sparse import csc_array,hstack,vstack
from scipy import sparse



class on_ADL() :
        
    def __init__(self,subset,family,penalty) :

        self.lr = 0.5
        self.lr_ldp = 0.5
        self.lam_list0 = [0.01]
        self.subset = subset
        self.R1 = 15
        self.R2 = 0.4
        self.T_list1 = [0] + [2 ** i * np.ceil(np.log(3644040))*10 for i in range(15)]
        self.T_list2 = [0] + [2 ** i * np.ceil(np.log(3644040))*10 for i in range(15)]
        self.family = family
        self.penalty = penalty
        self.shresh=0.5
        self.train_test_split=0.8
          
    # training         
    def fit( self, X_total,y_total,folder) :  
        epsilon = 1e-7
        self.NB = 83439
        self.B = 1
        self.p = 3644040
        self.h1=1
        self.t1=1
        self.h2=1
        self.t2=1
        self.R=float(self.R1)
        self.R_ldp=float(self.R2)
        self.folder_name=folder
        self.X_total=X_total
        self.y_total=y_total

        self.mu1 = epsilon * np.random.rand(self.p, 1) 
        self.theta1 = epsilon * np.random.rand(self.p, 1) 
        self.theta1_sum = np.zeros((self.p, 1))

        self.mu2 = epsilon * np.random.rand(self.p-1, len(self.subset))
        self.theta2 = epsilon * np.random.rand(self.p-1, len(self.subset))
        self.theta2_sum = np.zeros((self.p-1, len(self.subset)))

        self.beta_bar = np.zeros((self.p, 1))
        self.beta_tilde = np.zeros((self.p, 1))
        self.gamma_bar = np.zeros((self.p-1, len(self.subset)))
        self.gamma_tilde = np.zeros((self.p-1, len(self.subset)))
        self.beta_de = np.zeros((len(self.subset), 1))

        self.s1 = np.zeros((self.p))
        self.s2 = np.zeros((self.p, len(self.subset)))
        self.s3 = np.zeros((len(self.subset), 1))
        self.s4 = np.zeros((len(self.subset), 1))
        self.s5 = np.zeros((len(self.subset), 1))

        self.tao = np.zeros((len(self.subset), 1))
        self.lb = np.zeros((len(self.subset), 1))
        self.ub = np.zeros((len(self.subset), 1))

        self.zscore=stats.norm.ppf(.995)

        self.beta_trajec = np.zeros((len(self.subset),self.NB))
        self.debeta_trajec = np.zeros((len(self.subset),self.NB))
        self.lb_trajec = np.zeros((len(self.subset),self.NB))
        self.ub_trajec = np.zeros((len(self.subset),self.NB))
        self.tao_trajec = np.zeros((len(self.subset),self.NB))

        self.pred_err= []

        # Online training       
        for i in range( int(self.train_test_split*self.NB)) :

            self.i = i
            start_time = time.time()
            self.prox_step()
            print("---data %s---subset %s---%s seconds---"%((i+1),self.subset,(time.time()-start_time)))

            self.beta_trajec[:,i] = self.beta_bar[self.subset,0] #record reajectory
            self.debeta_trajec[:,i] = self.beta_de[:,0]
            self.lb_trajec[:,i] = self.lb[:,0]
            self.ub_trajec[:,i] = self.ub[:,0]
            self.tao_trajec[:,i] = self.tao[:,0]

            self.pred_err.append(self.Err())

            if self.i%50 == 0:
                self.plot1()
                self.plot_err()

        return self
      
    # RADAR step
    def prox_step( self ) :
      
        X = self.X_total[self.i, :]
        y = self.y_total[self.i].reshape(1,1)
  
        if self.i == self.T_list1[self.h1]  : 
            self.t1=1
            self.beta_bar= np.array(self.beta_tilde)
            self.theta1_sum = np.zeros((self.p,1))
            self.mu1 = np.zeros((self.p, 1))
            self.theta1 = np.array(self.beta_tilde)
            
            self.R=self.R1/(np.sqrt(2)**self.h1)
            self.h1=self.h1+1

        if self.i == self.T_list2[self.h2]  :
            self.t2=1

            self.gamma_bar= np.array(self.gamma_tilde) 
            self.theta2_sum = np.zeros((self.p-1,len(self.subset)))
            self.mu2 = np.zeros((self.p-1, len(self.subset)))
            self.theta2 = np.array(self.gamma_tilde)
            self.R_ldp=self.R2/(np.sqrt(2)**self.h2)
            self.h2=self.h2+1
        eta1 =(self.lr/np.power(self.t1,0.5))*((self.R1/self.R)**2)

        lam_list1 = [l*(self.R/self.R1) for l in self.lam_list0]
        eta2 = (self.lr_ldp/np.power(self.t2,0.5))*((self.R2/self.R_ldp)**2)
        
        lam_list2 = [l*(self.R_ldp/self.R2) for l in self.lam_list0]

        # cross-validating lambda
        err_list=[]
        for lam in lam_list1:
            mu1_cand,theta1_cand=radar.glm_RADAR(self.mu1, self.theta1, X, y,self.beta_bar, eta1, lam, self.R,self.family,self.penalty)
            theta1_sum_cand=self.theta1_sum + theta1_cand
            beta_cand=theta1_sum_cand/self.t1

            err_list.append(cal.negative_log_likelihood(y, X,  beta_cand, self.family))

        # choose lambda according to the negative log-likelihood
        lam1= lam_list1[np.argmin(err_list)]
        lam2= lam_list2[np.argmin(err_list)]

        self.mu1,self.theta1=radar.glm_RADAR(self.mu1, self.theta1, X, y,self.beta_bar, eta1, lam1, self.R,self.family,self.penalty)
        self.theta1_sum=self.theta1_sum + self.theta1
        self.beta_tilde=self.theta1_sum/self.t1

        # debias beta
        
        for k in range(len(self.subset)):
            
            r=self.subset[k]

            # low-dimensional projection
            o,u=radar.adaptive_RADAR(self.mu2[:,k], self.theta2[:,k], sparse.csr_array(np.delete(X.toarray(),[r],1)), csc_array(X.toarray()[:,r]) ,self.gamma_bar[:,k], eta2, lam2,self.R_ldp,
                                     self.penalty,cal.var_est(X,self.beta_bar,self.family))
            self.mu2[:,k]=np.reshape(o,(self.p-1,))
            self.theta2[:,k]=np.reshape(u,(self.p-1,))
            self.theta2_sum[:,k]=self.theta2_sum[:,k] + self.theta2[:,k]
            self.gamma_tilde[:,k]=self.theta2_sum[:,k]/self.t2

            if self.h2>=3:   #ignore the first two epochs

                gamma_k = np.insert(self.gamma_bar[:,k],r,-1)
                self.s1[k]=self.s1[k] + (X @ gamma_k).item()* (cal.mean_est(X,self.beta_bar,self.family) - y)
                self.s2[:,k] = self.s2[:,k] + (X @ gamma_k )*cal.var_est(X,self.beta_bar,self.family)*X
                self.s3[k] = self.s3[k] + (X @ gamma_k)*cal.var_est(X,self.beta_bar,self.family)*(X @ self.beta_bar)
                self.s4[k] = self.s4[k] + (X @ gamma_k)* X.toarray()[:,r].item() *cal.var_est(X,self.beta_bar,self.family)
                self.s5[k] = self.s5[k] + ((X @ gamma_k)**2) * ((cal.mean_est(X,self.beta_bar,self.family)-y)**2)
                
                self.tao[k] = np.sqrt(self.s5[k].item())/(-self.s4[k].item())
                    
                self.beta_de[k] = self.beta_bar[r] - ((self.s1[k]  +  (self.s2[:,k].T @ self.beta_bar).item() - self.s3[k].item())/self.s4[k].item())
                self.lb[k] = self.beta_de[k] - self.zscore * self.tao[k]
                self.ub[k] = self.beta_de[k] + self.zscore * self.tao[k]

        self.t1=self.t1+1
        self.t2=self.t2+1
    
        return self

    def parameters(self):

        return self.debeta_trajec,self.lb_trajec,self.ub_trajec,self.tao_trajec
    
    def Err(self):
        
        p=cal.sigmoid(self.X_total[int(self.train_test_split*self.NB+1):] @ self.beta_bar).reshape(-1)
        p[p >= self.shresh] = 1
        p[p < self.shresh] = 0
        return np.mean(np.abs(p.reshape(-1)-self.y_total[int(self.train_test_split*self.NB+1):].reshape(-1)))
    
    def plot1(self):
        tt=int(self.i+1)
        xx=np.arange(tt)
        fig = plt.figure(figsize=(6, 5*len(self.subset)))
        plt.plot(xx,self.debeta_trajec[0,0:tt].T)
        plt.plot(xx,self.beta_trajec[0,0:tt].T,alpha=0.5)
        plt.fill_between(xx, self.lb_trajec[0,0:tt].T, self.ub_trajec[0,0:tt].T,color='C0',alpha=0.2)
        plt.ylim (-2,2)
        plt.grid()
        plt.savefig(os.path.join(self.folder_name, "real-idx%s.png" % (self.subset))) 

    def plot_err(self):
        tt=int(self.i+1)
        xx=np.arange(tt)
        fig = plt.figure(figsize=(6, 5))
        plt.plot(xx,self.pred_err,color='C0')
        plt.ylim (0,0.7)
        plt.grid()
        plt.savefig(os.path.join(self.folder_name, "pred-idx%s.png" % (self.subset))) 
        