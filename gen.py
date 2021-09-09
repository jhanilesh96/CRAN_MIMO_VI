import os
import time
import numpy as np;
import tensorflow as tf;
import tensorflow_probability as tfp;
tfk = tf.keras;
tfd = tfp.distributions;
tf.keras.backend.set_floatx('float32');
from scipy.stats import unitary_group, cauchy, levy, t;
import cvxpy as cp 
np.random.seed(1233);
tf.random.set_seed(1233);
import matplotlib.pyplot as plt

use_gpu=1
if use_gpu>0:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def herm(X):
    return np.conj(X.T)

def randcomp(shape):
    return np.random.randn(*shape) + 1j*np.random.randn(*shape)

unitNMSE = lambda h_true, h_pred : np.mean(np.linalg.norm(h_true - h_pred)**2/np.linalg.norm(h_true)**2)
'''
-----------------------------------------------------------------
-----------------------------------------------------------------
-----------------------------------------------------------------
'''
import myutils

class GenData:
    def __init__(self, T=500, M=20, N=2, G=4, K=100, Ka=0.2, snrDB=20, P=40, Beta_range=[0.3,0.8], dgk=2, cgk=0.1, system=None, kwargs=None):
        self.params = myutils.Empty()
        self.kwargs = kwargs
        self.params.T = T
        self.params.M = M
        self.params.N = N
        self.params.G = G
        self.params.K = K
        self.params.Ka = Ka
        self.params.P = P
        self.params.Beta_range = Beta_range
        self.params.snr = snrDB
        self.params.dgk = dgk
        self.params.cgk = cgk
        self.system = self.initSystem() if system is None else system;
        ''' initialise variables '''
        Y = np.zeros([T,G,M,P], dtype='complex64')     # received signal
        B = np.zeros([T,G,K,M,N], dtype='complex64')    # channel, S*B
        H = np.zeros([T,G,K,M,N], dtype='complex64')    # channel, S*B
        ''' set User Activity '''
        D = self.setUserActivity()
        ''' set Sparsity Vector '''
        S = self.setSparsityVector() # latent sparsity vector
        ''' set B '''
        B[0] = dgk*randcomp([G,K,M,N])
        for t in range(1,T):
            B[t] = B[t-1] + cgk*randcomp([G,K,M,N])            
        '''set Channel and Received Signal'''
        for t in range(T):
            H[t] = B[t] * np.tile(np.expand_dims(S[t],-1), [1,1,1,N])
            ''' gkmn,knp->gmp, sum across repeated missing axis, 
                here n and k is missing; mn,np->mp is matmul; 
                it sums across k (user index), and axis g (RRH index) is present '''
            Y[t] = np.einsum('k,gkmn,knp->gmp',D[t],H[t],self.system.X)
            # for g in range(G):
            #     for k in range(K):
            #         Y[t,g] = Y[t,g] + D[t,k] * (H[t,g,k]@X[k])
        ''' add Noise '''
        W = randcomp(Y.shape)
        snr_curr = np.sum(np.abs(Y)**2)/np.sum(np.abs(W)**2);
        snr_required = 10**(snrDB/10);
        factor = snr_required/snr_curr;
        W = (factor**-0.5)*W
        self.B = B;
        self.H = H
        self.D = D
        self.S = S
        self.W = W;
        self.Y = Y+W;
    
    def getAlphaBeta(self):
        K = self.params.K
        Ka = self.params.Ka
        Beta_range = self.params.Beta_range
        betak = np.random.rand(K)*(Beta_range[1]-Beta_range[0]) + Beta_range[0]
        alphak = (Ka/(1-Ka))*betak
        return alphak, betak

    def initSystem(self):
        K = self.params.K
        N = self.params.N
        P = self.params.P
        Ka = self.params.Ka
        G = self.params.G
        Beta_range = self.params.Beta_range
        system = myutils.Empty()
        '''Pilot symbols, maximise orthogonality for each time instance'''
        X = np.zeros([K,N,P],dtype='complex64')
        for p in range(P):
            X[:,:,p] = unitary_group.rvs(max(K, N))[:K,:N]
        system.X = X;
        '''Betas'''
        system.alphak, system.betak = self.getAlphaBeta();
        ''' Set transition params - TGKM'''
        system.alpha_gm_01 = 0.10 * np.ones(G)
        system.alpha_gm_10 = 0.75 * np.ones(G)
        system.alpha_gt_01 = 0.05 * np.ones(G)
        system.alpha_gt_10 = 0.35 * np.ones(G)
        ''' a_g_mt'''
        system.alpha_g_111 = 0.8558 * np.ones(G)        # 0.9958
        system.alpha_g_011 = 0.4276 * np.ones(G)        # 0.3276
        system.alpha_g_101 = 0.2936 * np.ones(G)        # 0.3936
        system.alpha_g_001 = 0.0013 * np.ones(G)        # 0.0013
        ''' dependent transition variables '''
        system.alpha_gm_00_ = 1 - system.alpha_gm_01
        system.alpha_gm_11_ = 1 - system.alpha_gm_10
        system.alpha_gt_00_ = 1 - system.alpha_gt_01
        system.alpha_gt_11_ = 1 - system.alpha_gt_10
        system.alpha_g_110_ = 1 - system.alpha_g_111
        system.alpha_g_010_ = 1 - system.alpha_g_011
        system.alpha_g_100_ = 1 - system.alpha_g_101
        system.alpha_g_000_ = 1 - system.alpha_g_001
        return system

    def setUserActivity(self):
        T = self.params.T
        K = self.params.K
        D = np.zeros([T,K],dtype='bool')
        for k in range(K):
            D[:,k] = myutils.markov(self.system.alphak[k], self.system.betak[k], numStates=T)
        # if any time index has 0 active users, then redo a subset of users, repeat untill that is not the case
        while np.any(np.sum(D,axis=1) == 0):
            for _ in range(int(np.ceil(K/5))):
                k = np.random.randint(K)
                D[:,k] = myutils.markov(self.system.alphak[k], self.system.betak[k], numStates=T)
        
        return D;

    def setSparsityVector(self):
        T = self.params.T
        M = self.params.M
        N = self.params.N
        G = self.params.G
        K = self.params.K
        Ka = self.params.Ka
        P = self.params.P
        system = self.system;
        S = np.zeros([T,G,K,M], dtype='bool') # latent sparsity vector s = np.zeros([T,M], dtype='bool')
        ''' Helper function'''
        def setm(g,s,t):
            for m in range(1,M):
                if s[t, m-1]==1 and s[t-1, m]==1:   # 11
                        s[t,m] = np.random.rand()<system.alpha_g_111[g]
                if s[t, m-1]==0 and s[t-1, m]==1:   # 01
                        s[t,m] = np.random.rand()<system.alpha_g_011[g]
                if s[t, m-1]==1 and s[t-1, m]==0:   # 10
                        s[t,m] = np.random.rand()<system.alpha_g_101[g]
                if s[t, m-1]==0 and s[t-1, m]==0:   # 00
                        s[t,m] = np.random.rand()<system.alpha_g_001[g]
            return s;
        ''' for T=0/M=0, simple markov, parameters taken from Lian, L., Liu, A., & Lau, V. K. (2019). Exploiting dynamic sparsity for downlink FDD-massive MIMO channel tracking. IEEE Transactions on Signal Processing, 67(8), 2007-2021.'''
        for g in range(G):
            for k in range(K):
                s = np.zeros([T,M], dtype='bool')
                while np.sum(s[0,:])==0:
                    s[0,:] = myutils.markov(system.alpha_gm_01[g], system.alpha_gm_10[g], numStates=M)
                s[:,0] = myutils.markov(system.alpha_gt_01[g], system.alpha_gt_10[g], numStates=T, initState=s[0,0])
                for t in range(1,T):
                    s = setm(g,s,t);
                    while np.sum(s[t,:])==0:
                        s = setm(g,s,t)
                S[:,g,k,:] = s
                # s.astype('int')
        return S;

    def combine(self, data):
        assert np.allclose(self.X, data.X)
        # self.H = np.concatenate([self.H, data.H], axis=0);
        # self.D = np.concatenate([self.D, data.D], axis=0);
        # self.S = np.concatenate([self.S, data.S], axis=0);
        # self.W = np.concatenate([self.W, data.W], axis=0);
        # self.Y = np.concatenate([self.Y, data.Y], axis=0);


# T=10; M=4; N=2; G=25; K=100; Ka=0.2; snrDB=20; P=25; Beta_range=[0.3,0.8]; dgk=2; cgk=0.1
# S = np.zeros([T,G,K,M], dtype='bool') 
# s = np.zeros([T,M], dtype='bool')
# g = k = 0
# S[:,g,k,:].shape
# s.shape
if __name__ == '__main__':
    gData = GenData()
    myutils.save('gData',gData,folderName='data')
    gData2 = GenData(T=10, M=8, N=2, G=4, K=20, Ka=0.2, snrDB=20, P=40)
    myutils.save('gData_small',gData2,folderName='data')
    gData2 = GenData(T=5, M=4, N=2, G=1, K=20, Ka=0.2, snrDB=20, P=40)
    myutils.save('gData_Vsmall',gData2,folderName='data')
    '''visualise data'''
    T = gData.params.T
    M = gData.params.M
    N = gData.params.N
    G = gData.params.G
    K = gData.params.K
    Ka = gData.params.Ka
    P = gData.params.P
    X = gData.system.X
    system = gData.system
    H = gData.H
    D = gData.D
    S = gData.S
    W = gData.W
    Y = gData.Y
    # S.shape TGKM
    fig = plt.figure()
    _ = 0;
    for g in range(4):
        for k in range(6):
            _ = _ + 1
            plt.subplot(4, 6,_)
            plt.imshow(S[:50,g,k,:].T)
            plt.title('g = '+str(g)+', k = '+str(k))
            plt.xlabel('t')
            plt.ylabel('m')
    
    fig = plt.figure();
    plt.imshow(D[:50].T)
    plt.show()

'''
Y[t] = 0
for g in range(G):
    for k in range(K):
        Y[t,g] = Y[t,g] + D[t,k] * (H[t,g,k]@X[k])


np.allclose(Y[t], np.einsum('k,gkmn,knp->gmp',D[t],H[t],X))

'''


