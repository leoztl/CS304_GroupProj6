import os
import utils
import CDR
import mfcc
import CDR
from num2words import num2words
import numpy as np




def build_template():
    hmm_ls = []
    tails = []
    startNull = CDR.NullState()
    endNull = CDR.NullState()
    digits = [num2words(i) for i in range(10)]+['sil']
    for digit in digits:
        hmm = utils.load_hmm('./model/tz_trained', digit)
        hmm_ls.append(hmm)
        #print(len(hmm.Node_ls))
        startNull.next.append(hmm.Node_ls[0])
        tails.append(hmm.Node_ls[-1])
        hmm.Node_ls[-1].next = [endNull]
        hmm.Node_ls[0].edges[1] = (startNull,0.5)
        endNull.edges.append((hmm.Node_ls[-1], 1))

    #for tail in tails:
    #    print(tail.next)
    
    #startNull = hmm.Node_ls[0].edges[1][0]
    
    #utils.load_hmm("./model/tz/",)
    #endNull = CDR.appendWord(startNull, word)
    startNull.edges.append((endNull, 1))
    node_ls = CDR.flatten(startNull)
    #print(len(node_ls[0]))
    #print(startNull.next[0].next[0].next[0].next[0].next[0].next[0].next[0].name)
    return node_ls, tails
 
def parseBPT(bpt, tailBP):
    """
    Parse back pointer table obtained from DTW
    Return the recognition result and path
    """
    # initialize path
    seq = [None for i in range(len(bpt))]
    t = len(bpt) - 1
    #print(t)
    currNode = tailBP[-1]
    #print(currNode.name)
    currID = currNode.id
    seq[t] = currNode.name
    t -= 1
    
    while t >= 0:
        currNode = bpt[t][currID]
        #print(currNode.id)
        #print(t)
        #print(currNode.id)
        #try:
        currID = currNode.id
            #print(currNode.currDis)
        #except:
            #print(t)
            #print(currNode)
        seq[t] = currNode.name
        #print(seq[t])
        t -= 1
    # retrieve final results from the path
    currDigit = seq[0]
    digit_seq = [currDigit]
    for i in range(len(seq)):
        if seq[i] == currDigit:
            continue
        else:
            digit_seq.append(seq[i])
            currDigit = seq[i]
    # print(seq)
    return digit_seq, seq
 
 
 
def test(file_name, loopbackcost):   
    sentence = mfcc.mfcc_features(file_name, 40)
    node, tails= build_template()
    node_ls = node[0]
    nodeNum = node[1]
    
    startNull = node_ls[0]
    startNull.currDis = 0
    endNull = node_ls[-1]
    endNull.currDis = 0
    
    #for node in node_ls:
    #    print(node.isNull)

    tailBP = [None for i in range(len(sentence))]
    
    bpt = [[None for i in range(nodeNum)] for j in range(len(sentence))]
    for t in range(len(sentence)):
        vector = sentence[t]
        for currentNode in node_ls:
            currentNode.prevDis = currentNode.currDis
            
            if currentNode == endNull:
                minTail = tails[0]
                for n in tails:
                    if minTail.currDis > n.currDis:
                        minTail = n
                currentNode.currDis = minTail.currDis
                tailBP[t] = minTail
                continue

            if currentNode == startNull:
                currentNode.currDis = endNull.currDis
                continue
            
            if currentNode in startNull.next:
                if t == 0:
                    # at time 0, parentNode distance is 0
                    currentNode.currDis = currentNode.getDis(vector)
                    bpt[t][currentNode.id] = currentNode
                else:
                    # at other time t, can only self loop, parentNode is itself
                    if currentNode.currDis < startNull.currDis + loopbackcost:
                        currentNode.currDis = currentNode.prevDis + currentNode.getDis(vector)
                        bpt[t][currentNode.id] = currentNode
                    else:
                        currentNode.currDis = startNull.currDis + loopbackcost + currentNode.getDis(vector)
                        bpt[t][currentNode.id] = tailBP[t-1]
                continue
            parentDis = []
            for edge in currentNode.edges:
                parent = edge[0]
                transition = edge[1]
                if parent.isNull:
                    parentDis.append(parent.currDis - np.log(transition))
                else:
                    parentDis.append(parent.prevDis - np.log(transition))
            minIdx = np.argmin(parentDis)
            minParent = currentNode.edges[minIdx][0]
            distance = parentDis[minIdx]
            bpt[t][currentNode.id] = minParent
            currentNode.currDis = distance + currentNode.getDis(vector)
        #print(tailBP)
    #for node in tailBP:
    #    print(node.name)
    #print(node_ls[48].name)
    #print(node_ls[41].edges[1][0].name)
    return parseBPT(bpt, tailBP)[0]

#print(test('./sequences/tz/0123456789_2.wav', 100))
