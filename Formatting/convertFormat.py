import torch
import numpy as np
import math
from mpmath import mp

#predicted targets rmean rmin rmax

#The final format is
#idx lavel predict radius correct time

#R is a scaler (only one radius)
#returns log of volume
def computeV(n,R):
    a = mp.gamma(n/2+1)
    p = math.pi
    numer = (n/2)*np.log(p)
    den = mp.log(a)
    g = n*np.log(R)
    ans = numer-den+g
    return ans



#R is an array of radii
#returns log of volume
def computeV2(n,R):
    
    r = 0
    k = R.size
    for i in range(0,k):
        r = r + np.log(R[i])

    a = math.gamma(n/2+1)
    p = math.pi
    numerator = (n/2)*np.log(p)
    den = np.log(a)

    ans = numerator-den+r
    return ans;





def getVolumes():
    
    t = torch.load("Tensors_of_Radii.pt")
    b = t.numpy()
    c = b[:,0,:,:]
    print(c.shape)
    numIm = c.shape[0]
    Volumes = np.ones(numIm)
    for i in range(0,9664):
        curr = c[i,:,:]
        curr2 = curr.flatten()
        ans = computeV2(28,curr2)
        Volumes[i] = ans
        print(i)
    return Volumes

#this is for multi radii    
def format1():
    #R = getVolumes()
    f = open("Ours_data_updated.csv")
    #print(f.read())
    f2 = open("outputFile","w")
    print("idx\tlabel\tradius\tpredict\tcorrect\ttime",file=f2,flush=True)
    lines = f.readlines()
    idx = 0
    for line in lines:
        print(line)
        b = line.split(',')
        print(b)
        #r = R[idx-1]
        r = b[3] #r_mean
        #r = r_min.rstrip()
        label = b[1]
        prediction = b[0]
        correct = 0
        if label==prediction:
            correct=1
            
        if idx!=0:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(idx,label,r,prediction,correct,0),file=f2,flush=True)
        idx = idx+1




    f2.close()


#this is for single radii (cohen+mnist)
def format2():
    f = open("mnist_cohen.csv")
    #print(f.read())
    f2 = open("mnistCohenOut","w")
    print("idx\tlabel\tradius\tpredict\tcorrect\ttime",file=f2,flush=True)
    lines = f.readlines()
    idx = 0
    n = 28*28
    for line in lines:
        print(line)
        b = line.split(',')
        print(b)
        r = 0
        
        if idx!=0:
            r0 = b[3]
            r0 = r0.rstrip()
            r2 = float(r0)
            #r = computeV(n,r2)
            r = r0
        label = b[1]
        prediction = b[2]
        correct = 0
        if label==prediction:
            correct=1
            
        if idx!=0:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(idx,label,r,prediction,correct,0),file=f2,flush=True)
        idx = idx+1




    f2.close()



format1()
format2()


