"""
Taken from:
https://github.com/dionysos137/GBS_classicality/blob/master/approximate_sampling.py

as code for method in:
https://arxiv.org/pdf/1905.12075.pdf

with values from:
https://www.nature.com/articles/s41567-019-0567-8

Adapted with new values to get a more balanced input.
"""

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Thermal, Sgate, Interferometer
from scipy.stats import unitary_group, bernoulli

def ClosestClassicalState(r, K, eta,U,t):
    M = len(U)
    ap = eta * np.exp(2*r) + 1- eta
    an = eta * np.exp(-2 * r) + 1 - eta
    ss = 0.25 * np.log(ap/an)
    ns = 0.5 * (np.sqrt(ap * an) - 1)
    sc = 0.5 * np.log(2 * ns + 1)
    nt = 0.5 - 0.5 * np.sqrt(1 + 2 * t * np.sinh(2 * sc) * np.exp(2 * ss))
    s0= 0.5 * np.log((2*nt+1) / t)
    if ss < s0:
        st = ss
        nt = ns
    else:
        s0 = 0.5 * np.log((2 * nt + 1) / t)
        st = s0



    # build the state
    prog = sf.Program(M)
    with prog.context as q:
        for i in range(K):
            Thermal(nt) | q[i]
            Sgate(st)   | q[i]

        Interferometer(U)| q

    eng = sf.Engine('gaussian')
    result = eng.run(prog)
    cov = result.state.cov()
    #print(cov)
    return cov



def sampleGBS(r, K, eta, U, etaD, pD, e):
    M = len(U)
    tbar = 1 - 2 * pD / etaD

    term = np.log((1-2*pD/etaD) / (eta*np.exp(-2*r) + 1 - eta))
    if term > 0 and 1/np.cosh(0.5 * term) < np.exp(-e**2 / (4*K)):
        print("This experiment evades our algorithm!")
        return
    else:
        # sample from the output state
        mean = np.zeros([2*M])
        V = ClosestClassicalState(r, K, eta, U, tbar)
        import time
        t0 = time.time()
        x = np.random.multivariate_normal(mean, np.linalg.inv(V-tbar*np.identity(2*M)))
        # sample from the measurement
        #n = [bernoulli.rvs(np.exp(-(x[i]**2 + x[i+1]**2)/4)) for i in range(len(x)-1)] #Missing a few function params
        n = [bernoulli.rvs(((1-pD)/(1-etaD*(1-tbar)/2))*np.exp(-etaD*(x[i]**2 + x[i+1]**2)/4*(1-etaD*(1-tbar)/2))) for i in range(len(x)-1)]
        print("{} sec".format(time.time() - t0))
    return n

#values below from https://www.nature.com/articles/s41567-019-0567-8
#r = 0.1
#K = 4
#eta = 0.088
#U = unitary_group.rvs(12)
#etaD = 0.78
#pD = 10**(-4)
#e = 0.023

#New values to get some more balanced results for simulation, these give a more balanced output
r = 0.1
K = 4
eta = 0.5
U = unitary_group.rvs(12)
etaD = 0.78
pD = 10**(-1)
e = 0.023

n = sampleGBS(r, K, eta, U, etaD, pD, e)
print(n)