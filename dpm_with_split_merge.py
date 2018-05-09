
# coding: utf-8

# In[15]:



# coding: utf-8

# In[25]:



# coding: utf-8

# In[32]:



"""
Posterior sampling for Gaussian Mixture Model with CRP prior (DPGMM) using Gibbs sampler
Reference: https://pdfs.semanticscholar.org/9ece/0336316d78837076ef048f3d07e953e38072.pdf
"""
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
#from dists import NormalInverseWishart
from scipy.stats import gamma
from numpy import linalg as LA
import math


def fun1(K,zs_minus_i,sigma,mus,mvn,alpha,N,x):
    
    
    for k in range(K):
        nk_minus = zs_minus_i[zs_minus_i == k].shape[0]
        #print("nk_minus",nk_minus)
        crp = nk_minus / (N + alpha - 1)
        #print("crp ",crp)
        #lk=crp * mvn.pdf(x, mus[k], sigma)
        #probs.append(lk)
        probs[k] = crp * mvn.pdf(x, mus[k], sigma)
        #print("probabillity",probs[k])

    # Prob of creating new cluster
    crp = alpha / (N + alpha - 1)

    lik = mvn.pdf(x, mu0, sigma0+sigma)  # marginal dist. of xyu
    #hg=crp*lik
    #probs.append(hg)
    probs[K] = crp*lik
    return probs;

def fun2(K,zs_minus_i,sigma,mus,mvn,alpha,N,x,cluster):
    #nk_minus1=len(cluster1)
    #nk_minus2=len(cluster2)
    #print(nk_minus1,nk_minus2)
    probs=[]
    for k in range(K):
        nk_minus=len(cluster[k])
        #nk_minus = zs_minus_i[zs_minus_i == k].shape[0]
        #print("nk_minus",nk_minus)
        crp = nk_minus / (N + alpha - 1)
        #print("crp ",crp)
        lk=crp * mvn.pdf(x, mus[k], sigma)
        probs.append(lk)
        #probs[k] = crp * mvn.pdf(x, mus[k], sigma)
    '''crp = nk_minus / (N + alpha - 1)
    #print("crp ",crp1)
        probs[k] = crp * mvn.pdf(x, mus[k], sigma)
    print(probs[k])

    crp2 = nk_minus2 / (N + alpha - 1)
    #print("crp ",crp2)
    probs[4] = crp2 * mvn.pdf(x, mus[4], sigma)
    print(probs[4])'''

    # Prob of creating new cluster
    crp = alpha / (N + alpha - 1)
    #print("crp for new cluster",crp)
    lik = mvn.pdf(x, mu0, sigma0+sigma)  # marginal dist. of xyu
    hg=crp*lik
    probs.append(hg)
    #probs[K] = crp*lik
    return probs;

def kernel(X):
    s1=0
    s2=0
    for i in X:
        s1=s1+i[0]
        s2=s2+i[1]
        #print(i[0])
    mean_col1=s1/(len(X))
    mean_col2=s2/(len(X))
    mean=[[mean_col1,mean_col2]]
    col1=[]
    col2=[]
    for i in X:
        col1.append(i[0])
        col2.append(i[1])
    var1=np.var(col1)
    var2=np.var(col2)
    var=[[var1,var2]]
    v_var=np.var(var)
    v=[]
    for i in X:
        v1=math.exp(-(LA.norm(np.array(i) - np.array(mean)))/(2*v_var))
        v.append(v1)
        #print(v)
    return v   

# Generate data
X1 = np.random.multivariate_normal([2, 8], np.diag([0.5, 0.5]), size=40)
#print(X1)
#print("second\n")
X2 = np.random.multivariate_normal([10, 20], np.diag([0.5, 0.5]), size=50)
#print(X2)
#print("third\n")
X3 = np.random.multivariate_normal([15, 40], np.diag([0.5, 0.5]), size=30)
X4 = np.random.multivariate_normal([35, 10], np.diag([0.1, 0.1]), size=30)
X5 = np.random.multivariate_normal([35, 40], np.diag([0.3, 0.3]), size=30)
X6 = np.random.multivariate_normal([25, 37], np.diag([0.3, 0.3]), size=30)
#X3 = np.random.multivariate_normal([15, 40], np.diag([0.5, 0.5]), size=60)


from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import SVC
(X, y) = load_iris(return_X_y = True)
#(X, y) = load_digits(return_X_y = True)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pca1', 'pca2'])

X=principalDf[['pca1','pca2']].as_matrix()
#X=np.vstack([X])
X = np.vstack([X1, X2, X3,X4,X5,X6])
print("original data",len(X))

N, D = X.shape

# GMM params
mus = []  # List of 2x1 vector (mean vector of each gaussian)
sigma = np.eye(D)
prec = np.linalg.inv(sigma)  # Fixed precision matrix for all Gaussians
zs = np.zeros([N], dtype=int)  # Assignments
#print(zs)
C = []  # Cluster: binary matrix of K x M
Ns = []  # Count of each cluster

# CRP prior
alpha = 0.43

# Base distribution prior: N(mu0, prec0)
mu0 = np.ones(D)
sigma0 = np.eye(D)
prec0 = np.linalg.inv(np.eye(D))
G0 = st.multivariate_normal(mean=mu0, cov=np.eye(D))


# Initialize with ONE cluster
C.append(np.ones(N, dtype=int))
zs[:] = 0
Ns.append(N)
mus.append(G0.rvs())
print(mus)
K = 1

mvn = st.multivariate_normal




# Gibbs sampler
for it in range(10):
    # --------------------------------------------------------
    # Sample from full conditional of assignment from CRP prior
    # z ~ GEM(alpha)
    # --------------------------------------------------------
    cluster=[[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # For each data point, draw the cluster assignment
    
    for i in range(N):
        # Remove assignment from cluster
        # ------------------------------

        zi = zs[i]
        C[zi][i] = 0
        Ns[zi] -= 1

        # If empty, remove cluster
        if Ns[zi] == 0:
            # Fix indices
            zs[zs > zi] -= 1

            # Delete cluster
            del C[zi]
            del Ns[zi]
            del mus[zi]

            # Decrement cluster count
            K -= 1

        # Draw new assignment zi weighted by CRP prior
        # --------------------------------------------

        probs = np.zeros(K+1)
        zs_minus_i = zs[np.arange(len(zs)) != i]
        #print(zs_minus_i)
        # Probs of joining existing cluster
        probs=fun1(K,zs_minus_i,sigma,mus,mvn,alpha,N,X[i])

        # Normalize
        probs /= np.sum(probs)
        #print(probs)
        # Draw new assignment for i
        z = np.random.multinomial(n=1, pvals=probs).argmax()
        
        # Update assignment trackers
        if z == K:
            C.append(np.zeros(N, dtype=int))
            Ns.append(0)
            mus.append(G0.rvs())
            K += 1

        zs[i] = z
        C[z][i] = 1
        Ns[z] += 1
        
        #print(X[i][0])
        if z==1:
            plt.plot(X[i][0],X[i][1],'ro')
            cluster[1].append([X[i][0],X[i][1]])
        if z==2:
            plt.plot(X[i][0],X[i][1],'go')
            cluster[2].append([X[i][0],X[i][1]])
        if z==0:
            plt.plot(X[i][0],X[i][1],'yo')
            cluster[0].append([X[i][0],X[i][1]])
        if z==4:
            plt.plot(X[i][0],X[i][1],'bo')
            cluster[4].append([X[i][0],X[i][1]])
        if z==5:
            plt.plot(X[i][0],X[i][1],'co')
            cluster[5].append([X[i][0],X[i][1]])
        if z==3:
            plt.plot(X[i][0],X[i][1],'mo')
            cluster[3].append([X[i][0],X[i][1]])

    # -------------------------------------------------
    # Sample from full conditional of cluster parameter
    # -------------------------------------------------

    # Assume fixed covariance => posterior is Normal
    # mu ~ N(mu, sigma)
    for k in range(K):
        Xk = X[zs == k]
        Ns[k] = Xk.shape[0]

        # Covariance of posterior
        lambda_post = prec0 + Ns[k]*prec
        cov_post = np.linalg.inv(lambda_post)

        # Mean of posterior
        left = cov_post
        right = prec0 @ mu0 + Ns[k]*prec @ np.mean(Xk, axis=0)
        mus_post = left @ right

        # Draw new mean sample from posterior
        mus[k] = mvn.rvs(mus_post, cov_post)



for k in range(K):
    print('{} data in cluster-{}, mean: {}'.format(Ns[k], k, mus[k]))
#for i in range(1):
#    print(cluster[i])
#print(cluster[0])    
plt.show()
print('------------')
#print(len(cluster[0]))


# In[38]:

#X2=[[2.22042897, 4.40803534]]
X2=[[33.8862580752911088, 60.861739350630065]]
size_k=K
#print(X1[0][0],X1[0][1])

probs=fun2(K,zs_minus_i,sigma,mus,mvn,alpha,N+1,X2,cluster)
probs /= np.sum(probs)
z = np.random.multinomial(n=1, pvals=probs).argmax()
XX=np.vstack([cluster[z],X2])
ks=kernel(XX)
if ks[len(cluster[z])]>min(ks) and ks[len(cluster[z])]<max(ks):
    #cluster[z].append([X2])
    print('data is already exists')
    
    
else:
    cluster[z].append([X2[0][0],X2[0][1]])
    #print('probs of z and k')
    #print(probs[z],probs[K])
    #print('-----')
    F=(probs[z]*math.gamma(len(cluster[z])))/alpha*probs[K]
    #print("F value",F)

    # In[46]:

    X1=cluster[z]
    #print("lentgh")
    #print(len(X1))
    #cluster[z].append(X2)
    km=KMeans(n_clusters=2)
    km.fit(X1)
    new_c=[[],[]]
    for i in range(len(X1)):
        if km.labels_[i]==0:
            cluster[size_k].append([X1[i][0],X1[i][1]])
        elif km.labels_[i]==1:
            cluster[size_k+1].append([X1[i][0],X1[i][1]])
    print('length of k,i,j');
    print(len(cluster[z]))
    print(len(cluster[size_k]))
    print(len(cluster[size_k+1]))
    print('-------------')
    
        
    mus.append(G0.rvs())
    mus.append(G0.rvs())
    Ns.append(len(cluster[size_k]))
    Ns.append(len(cluster[size_k+1]))
    for k in range(size_k,size_k+2):
        #print('hello world')
        #print(k)
        lambda_post = prec0 + Ns[k]*prec
        cov_post = np.linalg.inv(lambda_post)

            # Mean of posterior
        left = cov_post
        right = prec0 @ mu0 + Ns[k]*prec @ np.mean(cluster[k], axis=0)
        mus_post = left @ right

            # Draw new mean sample from posterior
        mus[k] = mvn.rvs(mus_post, cov_post)

    #print(mus)
    probs = np.zeros(size_k+2)
    #print("zs before")
    #print(zs)
    zs.resize(1+N)
    #print(zs[0])


    zs[N]=z
    zs_minus_i = zs[np.arange(len(zs)) != N]

    probs=fun2(size_k+2,zs_minus_i,sigma,mus,mvn,alpha,N,X2,cluster)
    #print(probs)
    print(len(cluster[size_k]))

    #gamma_k=math.gamma(len(cluster[z]))
    #gamma_i=math.gamma(len(cluster[3]))
    #gamma_j=math.gamma(len(cluster[4]))
    #gamma_k=len(cluster[z])/N
    #gamma_j= len(cluster[4])/len(cluster[z])
    #gamma_i=len(cluster[3])/len(cluster[z])



    #for i in cluster[z]:
    #    print(i);
    probab=0;
    probab1=0;
    probab2=0;
    l=0;
    cnt=0
    for i in cluster[z]:
        X2=i
        probs=fun2(K,zs_minus_i,sigma,mus,mvn,alpha,N,X2,cluster)
        probs /= np.sum(probs)
        #z = np.random.multinomial(n=1, pvals=probs).argmax()
        #print('z value is =',z)
        #print(probs[z])
        cnt=cnt+1
        if probs[z]>10**(-3):
            probab=probab+probs[z]

        elif probs[z]<10**(-3):
            probab=probab+probab/(cnt)
        #print("after")
        #print(probs[z])

        #probab=probab+probs[z];
    print("---------------")
    print("kth")
    print(len(cluster[z]))
    #print(probab)
    print("ith")
    cnt=0
    print(len(cluster[size_k]))
    for i in cluster[size_k]:
        X2=i
        cnt=cnt+1;
        probs=fun2(size_k+2,zs_minus_i,sigma,mus,mvn,alpha,N,X2,cluster)
        probs /= np.sum(probs)
        #print('------------i-----')
        #print(probs[3])
        if probs[size_k]>10**(-3):
            probab1=probab1+probs[size_k]

        elif probs[size_k]<10**(-3):
            probab1=probab1+probab1/cnt
        #probab1=probab1+probs[3];
    #print(probab1)
    print("jth")
    print(len(cluster[size_k+1]))
    cnt=0
    for i in cluster[size_k+1]:
        X2=i
        cnt=cnt+1
        probs=fun2(size_k+2,zs_minus_i,sigma,mus,mvn,alpha,N,X2,cluster)
        probs /= np.sum(probs)
        #print('---------------j------')
        #print(probs[4])
        if probs[size_k+1]>10**(-3):
            probab2=probab2+probs[size_k+1]

        elif probs[size_k+1]<10**(-3):
            probab2=probab2+probab2/cnt
        #probab2=probab2+probs[4];
    #print("Final sum")
    #print(probab2)

    dpm_k=probab
    dpm_i=probab1
    dpm_j=probab2
    gamma_k=len(cluster[z])/N
    gamma_i=len(cluster[size_k])/len(cluster[z])
    gamma_j=len(cluster[size_k])/len(cluster[z])
    #print("probabilities")
    #print(dpm_k,dpm_i,dpm_j,gamma_k,gamma_i,gamma_j)
    rs=(gamma_k*dpm_k)/((alpha*gamma_i*dpm_i)*(gamma_j*dpm_j))
    #print("RS value")
    #print(rs)

    if rs<=1:
        del(cluster[z])
        size_k-=1
    elif rs>1:
        del(cluster[size_k])
        del(cluster[size_k+1])
    '''if rs<1:    
        for ii in range(size_k+1):
            print(len(cluster[ii]))
        #print(len(cluster[0]),len(cluster[1]),len(cluster[2]),len(cluster[3]))
    elif rs>1:
        for ii in range(size_k):
            print(len(cluster[ii]))
        #print(len(cluster[0]),len(cluster[1]),len(cluster[2]))'''
    if(rs<=1):
        probab=0
        rm=[]
        mini_1=999999999
        mini_2=999999999
        index_1=0
        index_2=0
        for i in range(K-1):
            cnt=0
            probab=0
            #print("checking cluster is ",i)
        #print('cluster i are ')
        #print(cluster[i])
            for j in range(size_k,size_k+2):
                if j==size_k:

                     #dpm of k
                    ls=[[],[],[]]
                    xx=cluster[i];
                    for m in range(len(xx)):
                        ls[0].append([xx[m][0],xx[m][1]])
                    xx=cluster[size_k]
                    for m in range(len(xx)):
                        ls[0].append([xx[m][0],xx[m][1]])

                    cnt=0
                    prob_join=0
                    for m in ls[0]:
                        X2=m
                        probs=fun2(1,zs_minus_i,sigma,mus,mvn,alpha,N,X2,ls)
                        probs /= np.sum(probs)
                    #print("prob_join-----")
                    #print(probs[0])
                        cnt=cnt+1
                        if probs[0]>10**(-3):
                            prob_join=prob_join+probs[0]

                        elif probs[0]<10**(-3):
                            prob_join=prob_join+prob_join/(cnt)
                #dpm of i
                    #print('prob_join==',j)
                    #print(prob_join)
                    cnt=0
                    probs_i=0
                    for j in cluster[i]:
                        X2=j
                        probs=fun2(K,zs_minus_i,sigma,mus,mvn,alpha,N,X2,cluster)
                        probs /= np.sum(probs)
                        cnt=cnt+1
                        if probs[i]>10**(-3):
                            probs_i=probs_i+probs[i]

                        elif probs[i]<10**(-3):
                            probs_i=probs_i+probs_i/(cnt)
                #gammas
                #print('prob_i===')
                #print(probs_i)
                    gamma_k_1=len(ls[0])/N
                    gamma_i_1=len(cluster[i])/len(ls[0])
                    gamma_j_1=len(cluster[size_k])/len(ls[0])
                #rm calculate    
                    rm_i=(alpha*gamma_i_1*gamma_j_1*probab1*probs_i)/(gamma_k_1*prob_join)
                    #print ("RM",rm_i)
                    for iii in range(size_k):
                        if iii==i:
                            if rm_i<mini_1:
                                index_1=iii
                                mini_1=rm_i
                    #if i==0:
                    #    if rm_i<mini_1:
                    #        mini_1=rm_i
                    #if i==1:
                    #    if rm_i<mini_1:
                    #        mini_1=rm_i
            
                elif j==size_k+1:
                #dpm j
                    cnt=0
                    probs_i=0
                    for j in cluster[i]:
                        X2=j
                        probs=fun2(K,zs_minus_i,sigma,mus,mvn,alpha,N,X2,cluster)
                        probs /= np.sum(probs)
                        cnt=cnt+1
                        if probs[i]>10**(-3):
                            probs_i=probs_i+probs[i]

                        elif probs[i]<10**(-3):
                            probs_i=probs_i+probs_i/(cnt)

                 #dpm of k
                    ls=[[],[]]
                    xx=cluster[i];
                    for m in range(len(xx)):
                        ls[0].append([xx[m][0],xx[m][1]])
                    xx=cluster[size_k+1]
                    for m in range(len(xx)):
                        ls[0].append([xx[m][0],xx[m][1]])

                    cnt=0
                    prob_join=0
                    for m in ls[0]:
                        X2=m
                        probs=fun2(1,zs_minus_i,sigma,mus,mvn,alpha,N,X2,ls)
                        probs /= np.sum(probs)
                    #print("-----")
                    #print(probs[0])
                        cnt=cnt+1
                        if probs[0]>10**(-3):
                            prob_join=prob_join+probs[0]

                        elif probs[0]<10**(-3):
                            prob_join=prob_join+prob_join/(cnt)
            
                #gammas
                    gamma_k_1=len(ls[0])/N
                    gamma_i_1=len(cluster[i])/len(ls[0])
                    gamma_j_1=len(cluster[size_k+1])/len(ls[0])
                #rm j
                    rm_j=(alpha*gamma_i_1*gamma_j_1*probab1*probs_i)/(gamma_k_1*prob_join)
                    #print ("RM =",rm_i)
                    
                    for iii in range(size_k):
                        if iii==i:
                            if rm_j<mini_2:
                                index_2=iii
                                mini_2=rm_j
        #print('------')
        #print("length",len(cluster[i]))
        #print(mini_1,mini_2)
          
            
    #print(len(cluster[0]))
        #print('rm=',mini_1)
        #print('rm=',mini_2)
        if mini_1<1:
            xx=cluster[size_k];
            for m in range(len(xx)):
                cluster[index_1].append([xx[m][0],xx[m][1]])
            del(cluster[size_k]) 

        if mini_2<1:
            xx=cluster[size_k+1];
            for m in range(len(xx)):
                cluster[index_2].append([xx[m][0],xx[m][1]])
            del(cluster[size_k+1])
        for iii in range (size_k+2):
            print('new cluster')
            print(len(cluster[iii]))
        #print(len(cluster[0]),len(cluster[1]),len(cluster[2]),len(cluster[3]))

    elif rs>1:
        for iii in range (size_k):
            print(len(cluster[iii]))
        #print(len(cluster[0]),len(cluster[1]),len(cluster[2]))

    




