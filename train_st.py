import DTW
import numpy as np
import utils
import CDR
import os
import sentence as st
from num2words import num2words


def get_recordings(folderpath):
    sentence_ls = []
    for file in os.listdir(folderpath):
        filename = os.path.join(folderpath, file)
        sentence = st.Sentence(filename)
        sentence_ls.append(sentence)
    return sentence_ls


def resetHead(hmm):
    head = hmm.getHead()
    new_edge = []
    for edge in head.edges:
        parentNode = edge[0]
        if parentNode == head:
            new_edge.append(edge)
    head.edges = new_edge


def resetTail(hmm):
    tail = hmm.getTail()
    tail.next = []


def connect(nullstate, hmm):
    resetHead(hmm)
    resetTail(hmm)
    nullstate.next.append(hmm.getHead())
    hmm.getHead().edges.append((nullstate, 1.0))
    new_null = CDR.NullState()
    hmm.getTail().next.append(new_null)
    new_null.edges.append((hmm.getTail(), 0.5))
    return new_null


def get_node_ls(sentence, obj_ls):
    sentencePath = sentence.name
    _, tail = os.path.split(sentencePath)
    digit_seq = tail.split(".")[0].split("_")[0]
    startNull = CDR.NullState()
    sil_0 = obj_ls[-1]
    currNull = connect(startNull, sil_0)
    for digit in digit_seq:
        hmm = obj_ls[int(digit)]
        currNull = connect(currNull, hmm)
    sil_1 = obj_ls[-2]
    currNull = connect(currNull, sil_1)
    return CDR.flatten(startNull)


def assign_vectors(sentence, path, feature_ls):
    for t in range(len(sentence.val)):
        vector = sentence.val[t]
        id = path[t].id
        vector_arr = feature_ls[id]
        vector_arr = np.vstack((vector_arr, vector))
        feature_ls[id] = vector_arr


def update(node_ls, feature_ls):
    for node in node_ls:
        if node.isNull:
            continue
        vector_arr = feature_ls[node.id]
        vectorNum = len(vector_arr)
        sentenceNum = 30
        # calculate self transition probability
        self_trans_prob = (vectorNum - sentenceNum) / vectorNum
        self_trans_prob = np.where(self_trans_prob == 0, np.finfo(np.float64).eps, self_trans_prob)
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
                if parentNode.isNull:
                    new_edges.append((parentNode, 1))
                else:
                    new_edges.append((parentNode, next_trans_prob))
        next_trans_prob = 1 - self_trans_prob

        node.edges = new_edges
        # update mean
        node.mean = mean
        # update covariance
        node.cov = cov_diag


def train(verbose):
    sequence_folder = "./sequences/tz"
    # read pre-trained isolated digit models
    obj_ls = []
    for i in range(10):
        obj = CDR.Hmm(num2words(i), "")
        obj_ls.append(obj)
    # read pre-trained silence model
    obj_ls.append(CDR.Hmm("sil", "sil0"))
    obj_ls.append(CDR.Hmm("sil", "sil1"))
    converge = False
    iteration = 0
    prev_cost = None
    sentence_ls = get_recordings(sequence_folder)
    while not converge:
        if verbose:
            print("iteration: ", iteration)
        total_cost = 0
        feature_ls = [np.empty((0, 39)) for i in range(73)]
        for sentence in sentence_ls:
            node_ls, nodeNum = get_node_ls(sentence, obj_ls)
            cost, bpt = DTW.DTW(sentence.val, node_ls)
            # print(bpt)
            total_cost += cost
            _, path = DTW.parseBPT(bpt)
            segment = sentence.update_segment(path)
            assign_vectors(sentence, path, feature_ls)
            if verbose:
                # utils.print_seq(path)
                print(segment)
                print("*" * 100)
        update(node_ls, feature_ls)
        if iteration != 0:
            converge = DTW.check_convergence(prev_cost, total_cost)

        iteration += 1
        prev_cost = total_cost

    # save models
    for obj in obj_ls:
        utils.save_hmm("./model/tz_trained", obj)


train(True)
