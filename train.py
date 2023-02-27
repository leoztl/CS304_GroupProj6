import numpy as np
import mfcc
import Sentence


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
    digit_seq = [currDigit[:-1]]
    for i in range(len(path)):
        if path[i].name == currDigit:
            continue
        else:
            digit_seq.append(path[i].name[:-1])
            currDigit = path[i].name

    return digit_seq, path


def load_mfcc(name):
    return


def update(node_ls, sentence_ls):
    """update transition probabilities, mean, covariance for all emitting node

    :param node_ls: node list
    :param sentence_ls: sentence list 
    """
    state_idx = 0
    for node in node_ls:
        if node.isNull:
            continue
        stateNum = 0
        vector_arr = np.empty((0, 39))
        for sentence in sentence_ls:
            stateNum += sentence.get_vectorNum(state_idx)
            state_vectors = sentence.getvectors(state_idx)
            vector_arr = np.vstack((vector_arr, state_vectors))
        # calculate self transition probability
        self_trans_prob = (stateNum - len(sentence_ls)) / stateNum
        # calculate mean
        mean = np.mean(vector_arr, 0)
        # calculate covariance matrix (diagonal)
        cov_mat = np.cov(vector_arr.T)
        cov_diag = np.diag(cov_mat)
        # update transition probabilities
        new_edges = []
        for edge in node.edges:
            parentNode = edge[0]
            if parentNode == node:
                new_edges.append((parentNode, self_trans_prob))
            else:
                new_edges.append((parentNode, 1 - self_trans_prob))
        node.edges = new_edges
        # update mean
        node.mean = mean
        # update covariance
        node.cov = cov_diag

        # move to next emitting state
        state_idx += 1


def check_convergence(prev_cost, curr_cost):
    if np.abs(prev_cost - curr_cost) / prev_cost < 0.01:
        return True
    else:
        return False


def train_sentence(name, node_ls):
    """train model for single sequence

    :param name: sequence name
    :param node_ls: pre-flattened graph
    :return: trained model
    """
    sentence_ls = load_mfcc(name)
    converge = False
    iteration = 0
    prev_cost = None
    while not converge:
        total_cost = 0
        for sentence in sentence_ls:
            cost, bpt = DTW(sentence, node_ls)
            total_cost += cost
            _, path = parseBPT(bpt)
            sentence.update_segment(path)
        update(node_ls, sentence_ls)
        if iteration != 0:
            converge = check_convergence(prev_cost, total_cost)
        prev_cost = total_cost
    return node_ls


def main():
    node_ls = []
    return


if '__name__' == '__main__':
    main()
