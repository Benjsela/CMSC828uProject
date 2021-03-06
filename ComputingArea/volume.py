import numpy as np
import math







#R is a scaler (only one radius)
#returns log of volume
def computeV(n,R):
    a = math.gamma(n/2+1)
    p = math.pi
    numer = (n/2)*np.log(p)
    den = np.log(a)
    g = n*np.log(R)
    ans = numer-den+g
    return ans



#R is an array of radii
#returns log of volume
def computeV2(n,R):
    k = n*n;
    r = 0
    for i in range(0,k):
        r = r + np.log(R[i])

    a = math.gamma(n/2+1)
    p = math.pi
    numerator = (n/2)*np.log(p)
    den = np.log(a)

    ans = numerator-den+r
    return ans;


curr = 1.5
inc = .01
for i in range(0,20):
    curr = curr+inc
    a = computeV(28,curr)
    print(a)




