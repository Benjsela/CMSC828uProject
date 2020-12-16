


#predicted targets rmean rmin rmax

#The final format is
#idx lavel predict radius correct time





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
    r_min = r_min.rstrip()
    label = b[1]
    prediction = b[0]
    correct = 0
    if label==prediction:
        correct=1

    if idx!=0:
        print("{}\t{}\t{}\t{}\t{}\t{}".format(idx,label,r_min,prediction,correct,0),file=f2,flush=True)
    idx = idx+1




f2.close()








