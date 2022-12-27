#Metodo de alias prueba
import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt

def alias(x, n , prob ):
    
    a = np.zeros(len(x))
    
    q = prob*len(x) #npi
    
    con = q < 1
    
    high = np.where(con == False)[0]
    
    low = np.where(con)[0]
    
    while len(high) and len(low) != 0:
        
        l = low[0]
        
        h = high[0]
        
        a[l] = h
        
        q[h] = q[h] - (1 - q[l])
        
        if q[h] < 1 :
            
            high = list(high)
            
            high.pop(0)
            
            low[0] = h
            
        else:
            
            low = list(low)
            
            low.pop(0)
            
    v = np.random.uniform(0, 1, n)
    
    u = np.random.uniform(0, 1, n)
    
    i = np.floor(u*(len(x))) #+ 1
    
    i = [round(y) for y in i]
    
    a = [round(t) for t in a]
    
    X = np.zeros(len(u))
    
    #X[0:len(x)] = x
    
    for j in range(len(u)):
        
        if v[j] < q[i[j]]:
            
            X[j] = x[i[j]]
        
        else:
            
            X[j] = x[a[i[j]]]

    return X

n = 20
p = 0.5
x = [i for i in range(n+1)]
fmp = ss.poisson(n*p).pmf(x)
n = 10**5
#np.random.seed(1)
rx = alias(x, n, fmp)
uni = np.unique(rx, return_counts=True)
plt.bar(uni[0], uni[1]/len(rx))
plt.scatter(x, fmp)
plt.show()


def alias_complex(M_conex)