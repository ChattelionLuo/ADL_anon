import numpy as np
from scipy import stats
import cal
import radar

class on_ADL() :
        
    def __init__(self,subset,family,beta0,penalty) :

        self.beta0 = beta0
        self.p = self.beta0.shape[0]
        self.lr = 10
        self.lam_list0 = list(np.logspace(-4, -2, 5, base=10))
        self.subset = subset
        self.R1 = 1.2*np.linalg.norm(beta0, ord=1)
        self.R2 = 1
        self.T_list1 = [0] + [np.ceil(2 ** i * np.ceil(np.log(self.p))*4) for i in range(15)]
        self.T_list2 = [0] + [np.ceil(2 ** i * np.ceil(np.log(self.p))*4)+1 for i in range(15)]
        self.family = family
        self.penalty = penalty
        
          
    # training         
    def fit( self, X_total,y_total) :   
        self.X_total = X_total
        self.y_total = y_total
        self.NB = X_total.shape[0]
        self.B = X_total.shape[1]
        
        self.h1=1
        self.t1=1
        self.h2=1
        self.t2=1
        self.R=float(self.R1)
        self.R_ldp=float(self.R2)

        self.mu1 = np.zeros((self.p, 1))
        self.theta1 = np.zeros((self.p, 1))#np.random.uniform(-0.00001, 0.00001, (self.p, 1))# for poisson regression
        self.theta1_sum = np.zeros((self.p, 1))

        self.mu2 = np.zeros((self.p-1, len(self.subset)))
        self.theta2 = np.zeros((self.p-1, len(self.subset)))
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
        self.s6 = np.zeros((len(self.subset), 1))

        self.tao = np.zeros((len(self.subset), 1))
        self.lb = np.zeros((len(self.subset), 1))
        self.ub = np.zeros((len(self.subset), 1))

        self.zscore=stats.norm.ppf(.975)

        self.beta_trajec = np.zeros((len(self.subset),self.NB))
        self.debeta_trajec = np.zeros((len(self.subset),self.NB))
        self.lb_trajec = np.zeros((len(self.subset),self.NB))
        self.ub_trajec = np.zeros((len(self.subset),self.NB))
        self.tao_trajec = np.zeros((len(self.subset),self.NB))

        self.opt= []  
        
        # Online training       
        for i in range( self.NB ) :

            self.i = i
            self.prox_step()

            #save result
            self.beta_trajec[:,i] = self.beta_bar[self.subset,0]
            self.debeta_trajec[:,i] = self.beta_de[:,0]
            self.lb_trajec[:,i] = self.lb[:,0]
            self.ub_trajec[:,i] = self.ub[:,0]
            self.tao_trajec[:,i] = self.tao[:,0]

        return self
      
    # RADAR step
    def prox_step( self ) :
      
        X=self.X_total[self.i,:,:] # New data
        y=self.y_total[self.i,:]
  
        if self.i == self.T_list1[self.h1]  : #Initialize new RADAR epoch
            self.t1=1
            self.beta_bar= np.array(self.beta_tilde)
            self.theta1_sum = np.zeros((self.p,1))
            self.mu1 = np.zeros((self.p, 1))
            self.theta1 = np.array(self.beta_tilde)
            
            self.R=self.R1/(np.power(2,0.5)**self.h1)
            self.h1=self.h1+1

        if self.i == self.T_list2[self.h2]  : # initialize new adaptive RADAR epoch
            self.t2=1

            self.gamma_bar= np.array(self.gamma_tilde)
            self.theta2_sum = np.zeros((self.p-1,len(self.subset)))
            self.mu2 = np.zeros((self.p-1, len(self.subset)))
            self.theta2 = np.array(self.gamma_tilde)
            
            self.R_ldp=self.R2/(np.power(2,0.5)**self.h2)
        
            self.tao_star=self.tao
            self.h2=self.h2+1

        eta1 = (self.lr/np.power(self.t1,0.5))*((self.R1/self.R)**2)
        lam_list1 = [l*(self.R/self.R1) for l in self.lam_list0]
        
        eta2 = (self.lr/np.power(self.t2,0.5))*((self.R2/self.R_ldp)**2)
        lam_list2 = [l*(self.R_ldp/self.R2) for l in self.lam_list0]

        err_list=[]

        # cross-validating lambda
        for lam in lam_list1:
            mu1_cand,theta1_cand=radar.glm_RADAR(self.mu1, self.theta1, X, y,self.beta_bar, eta1, lam, self.R,self.family,self.penalty)
            theta1_sum_cand=self.theta1_sum + theta1_cand
            beta_cand=theta1_sum_cand/self.t1

            err_list.append(cal.negative_log_likelihood(y, X,  beta_cand, self.family))
            

        # choose lambda according to the negative log-likelihood
        lam1=lam_list1[np.argmin(err_list)]
        lam2=lam_list2[np.argmin(err_list)]

        self.mu1,self.theta1=radar.glm_RADAR(self.mu1, self.theta1, X, y,self.beta_bar, eta1, lam1, self.R,self.family,self.penalty)
        self.theta1_sum=self.theta1_sum + self.theta1
        self.beta_tilde=self.theta1_sum/self.t1
        
        # node-wise lasso
        for k in range(len(self.subset)):
            
            r=self.subset[k]

            # adaptive_RADAR (online nodewise Lasso)
            o,u=radar.adaptive_RADAR(self.mu2[:,k], self.theta2[:,k], np.delete(X,[r],1), X[:,r] ,self.gamma_bar[:,k], eta2, lam2, self.R_ldp,self.penalty,cal.var_est(X,self.beta_bar,self.family))
            self.mu2[:,k]=np.reshape(o,(self.p-1,))
            self.theta2[:,k]=np.reshape(u,(self.p-1,))
            self.theta2_sum[:,k]=self.theta2_sum[:,k] + self.theta2[:,k]
            self.gamma_tilde[:,k]=self.theta2_sum[:,k]/self.t2

            # approximated debiasing step
            if self.h2>=3:

                gamma_k = np.insert(self.gamma_bar[:,k],r,-1)
                self.s1[k]=self.s1[k] + (X @ gamma_k).item()* (cal.mean_est(X,self.beta_bar,self.family) - y)
                self.s2[:,k] = self.s2[:,k] + (X @ gamma_k )*cal.var_est(X,self.beta_bar,self.family)*X
                self.s3[k] = self.s3[k] + (X @ gamma_k)*cal.var_est(X,self.beta_bar,self.family)*(X @ self.beta_bar)
                self.s4[k] = self.s4[k] + (X @ gamma_k)* X[:,r].item() *cal.var_est(X,self.beta_bar,self.family)
                self.s5[k] = self.s5[k] + ((X @ gamma_k)**2) * ((cal.mean_est(X,self.beta_bar,self.family)-y)**2)
                
                self.tao[k] = np.sqrt(self.s5[k].item())/(-self.s4[k].item()) #std of beta_de 
                self.beta_de[k] = self.beta_bar[r] -((self.s1[k]  +  (self.s2[:,k].T @ self.beta_bar).item() - self.s3[k].item())/self.s4[k].item())
                self.lb[k] = self.beta_de[k] - self.zscore * self.tao[k]   # confidence interval
                self.ub[k] = self.beta_de[k] + self.zscore * self.tao[k]

        self.t1=self.t1+1
        self.t2=self.t2+1
    
        return self