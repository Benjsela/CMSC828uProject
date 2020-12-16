import torch
import numpy as np
import math

#predicted targets rmean rmin rmax

#The final format is
#idx lavel predict radius correct time

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

    
def format1():
    R = getVolumes()
    f = open("ours_result_data.csv")
    #print(f.read())
    f2 = open("outputFile","w")
    print("idx\tlabel\tradius\tpredict\tcorrect\ttime",file=f2,flush=True)
    lines = f.readlines()
    idx = 0
    for line in lines:
        print(line)
        b = line.split(',')
        print(b)
        r_min = b[4]
        r = R[idx-1]
        r_min = r_min.rstrip()
        label = b[1]
        prediction = b[0]
        correct = 0
        if label==prediction:
            correct=1
            
            if idx!=0:
                print("{}\t{}\t{}\t{}\t{}\t{}".format(idx,label,r,prediction,correct,0),file=f2,flush=True)
                idx = idx+1




        f2.close()





format1()


