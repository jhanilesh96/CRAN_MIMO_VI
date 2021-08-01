# sim_fast is a boolean key for faster python implementation using einsum.



'''' A. Define Posterior Variables '''
'''' A1.1 Channel Gain'''
qB_mu = dgk*randcomp([T,G,K,M,N]) # np.ones([T,G,K,M,N], dtype='complex64')
qB_sigma = np.zeros([T,G,K,M,N,N], dtype='complex')
qB_sigma[:,:,:,:] = np.eye(N)
'''' A1.2 Channel Activity '''
qS = 0.2*np.ones([T,G,K,M], dtype='float')
'''' A1.3 Delta'''
qD = 0.2*np.ones([T,K],dtype='float')
'''' A1.4 Noise level'''
qSigma2_alpha = qSigma2_alpha0 * np.ones([T, G], dtype='float')
qSigma2_beta = qSigma2_beta0 * np.ones([T, G], dtype='float')
qH = np.zeros(qB_mu.shape, dtype='complex')

'''' A2 Pre computed variables'''
AK = np.zeros([K,N,N], dtype='complex')
for k in range(K):
    AK[k] = np.einsum('np,Np->nN', np.conj(X[k]), X[k])


def post1_ChannelGain(t,g,k,m):
    ns2inv = qSigma2_alpha[t,g]/qSigma2_beta[t,g];
    qH = postH(t,g)

    _Sigma = np.zeros([N,N], dtype='complex')
    if not sim_spatio_temporal:
        _Sigma += 1e-10 * AK[k]    
    _Sigma += (0.5*ns2inv) * qD[t,k] * qS[t,g,k,m] * AK[k]
    if sim_spatio_temporal:
        if t>0:
            if qS[t-1,g,k,m] > sim_thresh:
                _Sigma += (0.5/cgk**2) * np.eye(N)
            else:
                _Sigma += (0.5/dgk**2) * np.eye(N)
        else:
            _Sigma += (0.5/dgk**2) * np.eye(N)
        if sim_future:
            if t<T-1:
                if qS[t+1,g,k,m] > sim_thresh:
                    _Sigma += (0.5/cgk**2) * np.eye(N)
    Sigma = np.linalg.inv(2 * _Sigma)

    _mu = (0.5*ns2inv) * qD[t,k] * qS[t,g,k,m] * np.einsum('np,p->n', np.conj(X[k]), Y[t,g,m])
    if K > 1:
        if sim_fast:
            _mu -= (0.5*ns2inv) * qD[t,k] * qS[t,g,k,m] *\
                np.einsum('np,p->n', np.conj(X[k,:,:]),\
                    (np.einsum('knp,kn->p',X[:,:,:], qH[t,g,:,m]) - np.einsum('np,n->p',X[k,:,:], qH[t,g,k,m])) )
        else:
            for p in range(P):
                for j in range(K):
                    if j != k:
                        _mu -= (0.5*ns2inv) * qD[t,k] * qS[t,g,k,m] * np.conj(X[k,:,p]) * \
                            np.dot(X[j,:,p], qD[t,j] * qS[t,g,j,m] * qB_mu[t,g,j,m])
    
    if sim_spatio_temporal:
        if t>0:
            if qS[t-1,g,k,m] > sim_thresh:
                _mu += (0.5/cgk**2) * qB_mu[t-1,g,k,m]
        if sim_future:
            if t<T-1:
                if qS[t+1,g,k,m] > sim_thresh:
                    _mu += (0.5/cgk**2) * qB_mu[t+1,g,k,m]    
        
    mu = Sigma@(2*_mu)
    Update_LocVariables(idx=1, args=[t,g,k,m], newVariables=[mu, Sigma])


'''' B2. Channel Activity'''
# Y : TGMP
# B,H : TGKMN
# D : TK
# S : TGKM
# X : KNP
def post2_ChannelActivity(t,g,k,m):
    ns2inv = qSigma2_alpha[t,g]/qSigma2_beta[t,g];
    A = AK[k]
    qH = postH(t,g)
    ss_m0t0 = system.alpha_gm_01[g]/(system.alpha_gm_01[g] + system.alpha_gm_10[g])
    
    _s = 0j
    _s += (0.5*ns2inv) * qD[t,k] * np.dot(qB_mu[t,g,k,m],\
         np.einsum('p,np->n', np.conj(Y[t,g,m]), X[k]))
    _s += (0.5*ns2inv) * qD[t,k] * np.dot(np.conj(qB_mu[t,g,k,m]),\
         np.einsum('p,np->n', Y[t,g,m], np.conj(X[k])))
    _s -= (0.5*ns2inv) * qD[t,k] * (np.conj(qB_mu[t,g,k,m])@A@qB_mu[t,g,k,m].T + np.trace(A@qB_sigma[t,g,k,m]))
    if K > 1:
        if sim_fast:
            _s -= (0.5*ns2inv) * qD[t,k] * np.einsum('p,p->', np.einsum('n,np->p',qB_mu[t,g,k,m], X[k,:,:]), \
                np.conj(np.einsum('kn,knp->p',qH[t,g,:,m], X[:,:,:]) - np.einsum('n,np->p',qH[t,g,k,m], X[k,:,:])))
            _s -= (0.5*ns2inv) * qD[t,k] * np.einsum('p,p->', np.einsum('n,np->p',np.conj(qB_mu[t,g,k,m]), np.conj(X[k,:,:])), \
                (np.einsum('kn,knp->p',qH[t,g,:,m], X[:,:,:]) - np.einsum('n,np->p',qH[t,g,k,m], X[k,:,:])))
        else:
            for p in range(P):
                for j in range(K):
                    if k != j:
                        _s -=  (0.5*ns2inv) * qD[t,k] * np.dot(qB_mu[t,g,k,m], X[k,:,p]) * \
                            np.conj(qD[t,j] * qS[t,g,j,m] * np.dot(qB_mu[t,g,j,m], X[j,:,p]));
                        _s -=  (0.5*ns2inv) * qD[t,k] * np.dot(np.conj(qB_mu[t,g,k,m]), np.conj(X[k,:,p])) * \
                            qD[t,j] * qS[t,g,j,m] * np.dot(qB_mu[t,g,j,m], X[j,:,p]);
    
    # prior, t-1 and m-1
    if t==0 or (not sim_spatio_temporal):
        if m==0 or (not sim_spatio_temporal):
            _s += logit(ss_m0t0)
        else:
            _s += (1-qS[t,g,k,m-1])*logit(system.alpha_gm_01[g])
            _s -= qS[t,g,k,m-1]*logit(system.alpha_gm_10[g])
    else:
        if m==0:
            _s += (1-qS[t-1,g,k,m])*logit(system.alpha_gt_01[g])
            _s -= qS[t-1,g,k,m]*logit(system.alpha_gt_10[g])
        else:
            _s += qS[t,g,k,m-1]*qS[t-1,g,k,m]*logit(system.alpha_g_111[g])
            _s += (1-qS[t,g,k,m-1])*qS[t-1,g,k,m]*logit(system.alpha_g_011[g])
            _s += qS[t,g,k,m-1]*(1-qS[t-1,g,k,m])*logit(system.alpha_g_101[g])
            _s += (1-qS[t,g,k,m-1])*(1-qS[t-1,g,k,m])*logit(system.alpha_g_001[g])

    # effect of next, m+1    
    if sim_spatio_temporal:
        if m!=M-1:
            if t==0:
                _s += qS[t,g,k,m+1]*(log(system.alpha_gm_11_[g]) - log(system.alpha_gm_01[g]))
                _s += (1-qS[t,g,k,m+1])*(log(system.alpha_gm_10[g]) - log(system.alpha_gm_00_[g]))
            else:
                _s += qS[t,g,k,m+1]*qS[t-1,g,k,m+1]*(log(system.alpha_g_111[g])-log(system.alpha_g_011[g]))
                _s += qS[t,g,k,m+1]*(1-qS[t-1,g,k,m+1])*(log(system.alpha_g_101[g])-log(system.alpha_g_001[g]))
                _s += (1-qS[t,g,k,m+1])*qS[t-1,g,k,m+1]*(log(system.alpha_g_110_[g])-log(system.alpha_g_010_[g]))
                _s += (1-qS[t,g,k,m+1])*(1-qS[t-1,g,k,m+1])*(log(system.alpha_g_100_[g])-log(system.alpha_g_000_[g]))

    if sim_spatio_temporal:
        if sim_future:
            if t<T-1:
                if m==0:
                    _s += qS[t+1,g,k,m] * (log(system.alpha_gt_11_[g])-log(system.alpha_gt_01[g]))
                    _s += (1-qS[t+1,g,k,m]) * (log(system.alpha_gt_10[g])-log(system.alpha_gt_00_[g]))
                else:
                    _s += qS[t+1,g,k,m]*qS[t+1,g,k,m-1]*(log(system.alpha_g_111[g])-log(system.alpha_g_101[g]))
                    _s += qS[t+1,g,k,m]*(1-qS[t+1,g,k,m-1])*(log(system.alpha_g_011[g])-log(system.alpha_g_001[g]))
                    _s += (1-qS[t+1,g,k,m])*(1-qS[t+1,g,k,m-1])*(log(system.alpha_g_010_[g])-log(system.alpha_g_000_[g]))
                    _s += (1-qS[t+1,g,k,m])*qS[t+1,g,k,m-1]*(log(system.alpha_g_110_[g])-log(system.alpha_g_100_[g]))
    
    assert np.allclose(np.real(_s), _s)
    Update_LocVariables(idx=2, args=[t,g,k,m], newVariables=[_s])


'''' B3. User Activity'''
# Y : TGMP
# B,H : TGKMN
# D : TK
# S : TGKM
# X : KNP
def post3_Delta(t,k):
    A = AK[k]
    qH = postH(t)
    d0 = system.alphak[k]/(system.alphak[k] + system.betak[k]);
    
    _d = 0j
    for g in range(G):
        ns2inv = qSigma2_alpha[t,g]/qSigma2_beta[t,g]; 
        if sim_fast:
            _d_ = 0;
            _d_ += (0.5*ns2inv) * np.einsum('mn,mn->',qB_mu[t,g,k]*np.expand_dims(qS[t,g,k],-1), np.einsum('mp,np->mn', np.conj(Y[t,g]), X[k]))
            _d_ += (0.5*ns2inv) * np.einsum('mn,mn->',np.conj(qB_mu[t,g,k])*np.expand_dims(qS[t,g,k],-1), np.einsum('mp,np->mn', Y[t,g], np.conj(X[k])))
            _d_ -= (0.5*ns2inv) * np.sum(qS[t,g,k] * (np.einsum('mn,nN,mN->m',np.conj(qB_mu[t,g,k]),A,qB_mu[t,g,k]) + np.einsum('nN,mnN->m',A, qB_sigma[t,g,k])))
            _d_ -= np.sum( (0.5*ns2inv) * np.expand_dims(qS[t,g,k,:],-1) * np.einsum('mn,np->mp',qB_mu[t,g,k,:], X[k,:,:]) *\
                np.conj(np.einsum('kmn,knp->mp',qH[t,g,:,:], X[:,:,:]) - np.einsum('mn,np->mp',qH[t,g,k,:], X[k,:,:])) ) 
            _d_ -= np.sum( (0.5*ns2inv) * np.expand_dims(qS[t,g,k,:],-1) * np.einsum('mn,np->mp',np.conj(qB_mu[t,g,k,:]), np.conj(X[k,:,:])) *\
                (np.einsum('kmn,knp->mp',qH[t,g,:,:], X[:,:,:]) - np.einsum('mn,np->mp',qH[t,g,k,:], X[k,:,:])) )
            _d += _d_
        else:
            for m in range(M):
                _d += (0.5*ns2inv) * qS[t,g,k,m] * np.dot(qB_mu[t,g,k,m],\
                    np.einsum('p,np->n', np.conj(Y[t,g,m]), X[k]))
                _d += (0.5*ns2inv) * qS[t,g,k,m] * np.dot(np.conj(qB_mu[t,g,k,m]),\
                    np.einsum('p,np->n', Y[t,g,m], np.conj(X[k])))
                _d -= (0.5*ns2inv) * qS[t,g,k,m] * (np.conj(qB_mu[t,g,k,m])@A@qB_mu[t,g,k,m].T + np.trace(A@qB_sigma[t,g,k,m]))
                if K > 1:
                    for p in range(P):
                        for j in range(K):
                            if k != j:
                                _d -=  (0.5*ns2inv) * qS[t,g,k,m] * np.dot(qB_mu[t,g,k,m], X[k,:,p]) * \
                                    np.conj(qD[t,j] * qS[t,g,j,m] * np.dot(qB_mu[t,g,j,m], X[j,:,p]));
                                _d -=  (0.5*ns2inv) * qS[t,g,k,m] * np.dot(np.conj(qB_mu[t,g,k,m]), np.conj(X[k,:,p])) * \
                                    qD[t,j] * qS[t,g,j,m] * np.dot(qB_mu[t,g,j,m], X[j,:,p]);
    
    # prior
    if t == 0 or (not sim_spatio_temporal): 
        _d += logit(d0)
    else:
        _d += (1-qD[t-1,k]) * logit(system.alphak[k])
        _d -= qD[t-1,k] * logit(system.betak[k])
    
    if sim_spatio_temporal:
        if sim_future:
            if t < T-1:
                _d += (1-qD[t+1,k]) * (log(system.betak[k]) - log(1-system.alphak[k]))
                _d += qD[t+1,k] * (log(1-system.betak[k]) - log(system.alphak[k]))

    assert np.allclose(np.real(_d), _d)
    Update_LocVariables(idx=3, args=[t,k], newVariables=[_d])
    

'''' B4. Noise Level'''
# Y : TGMP
# B,H : TGKMN
# D : TK
# S : TGKM
# X : KNP
def post4_NoiseLevel(t,g):
    qH = postH(t,g)
    _alpha = qSigma2_alpha0 + 0.5
    _beta = qSigma2_beta0 + 0.5 * np.linalg.norm(Y[t,g] - np.einsum('kmn,knp->mp', qH[t,g], X) )**2
    Update_LocVariables(idx=4, args=[t,g], newVariables=[_alpha, _beta])

    
def Update_LocVariables(idx, args, newVariables):
    if idx == 1:
        [t,g,k,m], [mu, Sigma] = args, newVariables
        qB_mu[t,g,k,m] = sim_rho * mu + (1 - sim_rho) * qB_mu[t,g,k,m]
        qB_sigma[t,g,k,m] = sim_rho * Sigma + (1 - sim_rho) * qB_sigma[t,g,k,m]
    elif idx == 2:
        [t,g,k,m], [_s] = args, newVariables
        qS[t,g,k,m] = sim_rho * expit(np.real(_s)) + (1 - sim_rho) * qS[t,g,k,m]
    elif idx == 3:
        [t,k], [_d] = args, newVariables
        qD[t,k] = sim_rho * expit(np.real(_d)) + (1 - sim_rho) * qD[t,k]
    elif idx == 4:
        [t,g], [_alpha, _beta] = args, newVariables
        qSigma2_alpha[t,g] = sim_rho * _alpha + (1 - sim_rho) * qSigma2_alpha[t,g]
        qSigma2_beta[t,g] = sim_rho * _beta + (1 - sim_rho) * qSigma2_beta[t,g]


