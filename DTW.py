import numpy as np


def DTW(sentence, node_ls):
    """Perform DTW between mfcc features from a sentence and flattened template tree, return final cost and backpointer table

    :param sentence: mfcc vectors
    :param node_ls: python list of nodes
    :return: final cost, backpointer table
    """
    nodeNum = len(node_ls)
    vectorNum = len(sentence)
    startNull = node_ls[0]
    startNull.currDis = 0
    bpt = [[None for i in range(nodeNum)] for j in range(vectorNum)]
    for t in range(vectorNum):
        vector = sentence[t]
        for currentNode in node_ls:
            currentNode.prevDis = currentNode.currDis
            if currentNode == startNull:
                continue
            if currentNode in startNull.next:
                if t == 0:
                    # at time 0, parentNode distance is 0
                    currentNode.currDis = currentNode.getDis(vector)
                    bpt[t][currentNode.id] = currentNode
                else:
                    # at other time t, can only self loop, parentNode is itself
                    currentNode.currDis = currentNode.prevDis + currentNode.getDis(vector)
                    bpt[t][currentNode.id] = currentNode
                continue
            parentDis = []
            for edge in currentNode.edges:
                parent = edge[0]
                transition = edge[1]
                if transition == 0:
                    print(currentNode.edges)
                if parent.isNull:
                    parentDis.append(parent.currDis - np.log(transition))
                else:
                    parentDis.append(parent.prevDis - np.log(transition))
            minIdx = np.argmin(parentDis)
            minParent = currentNode.edges[minIdx][0]
            if minParent.isNull:
                minParent = bpt[t - 1][minParent.id]
            distance = parentDis[minIdx]
            bpt[t][currentNode.id] = minParent
            """ print("currentNode: ", currentNode.name)
            print("minParent:", minParent.name) """
            if currentNode.isNull:
                currentNode.currDis = distance
            else:
                currentNode.currDis = distance + currentNode.getDis(vector)
    endNode = node_ls[-1]
    cost = endNode.currDis
    return cost, bpt


def parseBPT(bpt):
    """Parse back pointer table obtained from DTW. Return the recognition result and path

    :param bpt: backpointer table
    :return: results, path
    """
    # initialize path to hold state object
    path = [None for i in range(len(bpt))]
    t = len(bpt) - 1
    currNode = bpt[t][-1]
    # print("currNode: ", currNode.name)
    currID = currNode.id
    path[t] = currNode
    t -= 1
    while t >= 0:
        currNode = bpt[t][currID]
        # print(currNode.name)
        currID = currNode.id
        path[t] = currNode
        t -= 1
    # retrieve final results from the path
    currDigit = path[0].name
    # print(currDigit)
    digit_seq = [currDigit[:-1]]
    for i in range(len(path)):
        if path[i].name == currDigit:
            continue
        else:
            digit_seq.append(path[i].name[:-1])
            currDigit = path[i].name

    return digit_seq, path


def check_convergence(prev_cost, curr_cost, threshold):
    # print(prev_cost, curr_cost)
    if np.abs(prev_cost - curr_cost) / prev_cost < threshold:
        return True
    else:
        return False
