import numpy as np
import cal
import sys

def glm_RADAR(mu, theta, X, Y, beta_k, eta, lam, R_k,family,penalty):
    d = mu.shape[0]
    q1 = 2 * np.log(d) / (2 * np.log(d) - 1)
    q2 = 2 * np.log(d)

    Y=np.reshape(Y,(1,1))
    mu=np.reshape(mu,(X.shape[1],1))
    theta=np.reshape(theta,(X.shape[1],1))
    beta_k=np.reshape(beta_k,(X.shape[1],1))

    mu = mu + (X.T @ (cal.mean_est(X,theta,family) - Y)) + cal.regularizer(theta,lam,penalty)
    
    mu_q_norm = np.power(np.sum(np.power(np.abs(mu), q2)), 1 / q2)

    xi= np.maximum((eta * mu_q_norm * R_k * (q1 - 1)) - 1,0)

    theta = beta_k - ((np.power(mu_q_norm, 2 - q2) * R_k * R_k * eta * (q1 - 1) * np.power(np.abs(mu), q2 - 1) *
                       np.sign(mu)) / (xi + 1))

    mu=np.reshape(mu,(X.shape[1],1))
    theta=np.reshape(theta,(X.shape[1],1))

    return mu , theta


def adaptive_RADAR(mu, theta, X, Y, beta_k, eta, lam, R_k,penalty,Var):
    d = mu.shape[0]
    q1 = 2 * np.log(d) / (2 * np.log(d) - 1)
    q2 = 2 * np.log(d)

    #X=np.reshape(X,(1,X.shape[1]))
    Y=np.reshape(Y,(1,1))
    mu=np.reshape(mu,(X.shape[1],1))
    theta=np.reshape(theta,(X.shape[1],1))
    beta_k=np.reshape(beta_k,(X.shape[1],1))

    mu = mu + (X.T @ (X @ theta - Y)) * Var + cal.regularizer(theta,lam,penalty)

    mu_q_norm = np.power(np.sum(np.power(np.abs(mu), q2)), 1 / q2)

    xi= np.maximum((eta * mu_q_norm * R_k * (q1 - 1)) - 1,0)

    theta = beta_k - ((np.power(mu_q_norm, 2 - q2) * R_k * R_k * eta * (q1 - 1) * np.power(np.abs(mu), q2 - 1) *
                       np.sign(mu)) / (xi + 1))

    mu=np.reshape(mu,(X.shape[1],1))
    theta=np.reshape(theta,(X.shape[1],1))

    return mu , theta