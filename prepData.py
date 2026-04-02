import numpy as np
import os
nuc_to_idx="ATCG"


revComplDict="ACNGT"
def revCompl(seq):
    return "".join(revComplDict[ -1-revComplDict.index(x) ] for x in seq[::-1])

def DataToSeq(array):
    seq=[]
    for row in array:
        for idx,chr in enumerate(nuc_to_idx):
            if row[idx]==1:
                seq.append(chr)
                continue
            else: seq.append("N")
    return "".join(seq)


def Prep(filepath, dataX, dataChr, includeOtherStrand=False, includeUnpairProb=False):
    with open(filepath, "r") as f:
        headers = f.readline()
        # print(headers)
        counter = 0

        for line in f:
            try:
                chr, seq, prob = line.strip(" \n").split("\t")
                data = np.zeros((5, 100) if includeUnpairProb else (4, 100), np.float32)
                counter += 1
                for idx, nuc in enumerate(seq):
                    if nuc == "N":
                        data[0][idx] = 0.25
                        data[1][idx] = 0.25
                        data[2][idx] = 0.25
                        data[3][idx] = 0.25
                    else:
                        data[nuc_to_idx.index(nuc)][idx] = 1.0  # probable error cause. unknown character
                    # else: print(filepath,counter,nuc)
                if includeUnpairProb:
                    prob = prob.split(",")
                    prob = [float(x) for x in prob]  # probable error cause. non digit character
                    data[4] = prob

                dataX.append(data)
                dataChr.append(chr)

                if includeOtherStrand:
                    seq = revCompl(seq)
                    data = np.zeros((5, 100) if includeUnpairProb else (4,100), np.float32)
                    for idx, nuc in enumerate(seq):
                        if nuc == "N":
                            data[0][idx] = 0.25
                            data[1][idx] = 0.25
                            data[2][idx] = 0.25
                            data[3][idx] = 0.25
                        else:
                            data[nuc_to_idx.index(nuc)][idx] = 1.0
                        # else: print(filepath,counter,nuc)
                    if includeUnpairProb:
                        prob = prob[::-1]
                        data[4] = prob

                    dataX.append(data)
                    dataChr.append(chr)
            except:
                print("ERROR:", line[:1000])
                print("ERROR: chr:", chr, "seq:", seq, "prob:", prob)
                print("ERROR: seq chars:", set(seq), "prob chars:", set(prob))
                print("ERROR in file:", filepath, "at line", counter)
                continue
        # print(dataX[0])

def PrepDir(directory, defaultY=1.0, maxFilesToProcess=-1, includeOtherStrand=False, includeUnpairProb=False):
    X=[]
    C=[]
    fileCount=0
    for filename in os.listdir(directory):
        if (fileCount==maxFilesToProcess): break
        filepath = os.path.join(directory, filename)
        print(filename)
        if (not os.path.isfile(filepath)) or (not filename.endswith(".c")):
            continue
        fileCount += 1
        Prep(filepath, X, C, includeOtherStrand, includeUnpairProb)

    return np.array(X),np.array([defaultY for x in X]),C


if __name__=="__main__":
    x,y=PrepDir("/home/osman/PycharmProjects/TFproject/negcont.c", defaultY=0.0, includeOtherStrand=True)
    print(x)
    print(y)

