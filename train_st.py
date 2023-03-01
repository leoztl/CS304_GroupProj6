import DTW
import numpy as np
import utils
import CDR
import os
import sentence as st
from time import time
from num2words import num2words


def get_recordings(folderpath):
    """read and return all recordings under folderpath

    :param folderpath: path of recording folder
    :return: list of sentence object
    """
    sentence_ls = []
    for file in os.listdir(folderpath):
        filename = os.path.join(folderpath, file)
        sentence = st.Sentence(filename)
        sentence_ls.append(sentence)
    return sentence_ls


def resetHead(hmm):
    """
    remove null state of last alignment from edges of head state of given hmm
    """
    head = hmm.getHead()
    new_edge = []
    for edge in head.edges:
        parentNode = edge[0]
        # ignore all non-emitting parents
        if not parentNode.isNull:
            new_edge.append(edge)
    head.edges = new_edge


def resetTail(hmm):
    """
    remove null state of last alignment from children of tail state of given hmm
    """
    tail = hmm.getTail()
    tail.next = []


def connect(nullstate, hmm):
    """
    first remove all null state from last alignment,
    then connect hmm after give null state and connect a new null state after the hmm, return the new null state

    :param nullstate: given nullstate
    :param hmm: given hmm
    :return: new initizlied nullstate
    """
    # remove all non-emitting states from last alignment
    resetHead(hmm)
    resetTail(hmm)
    # connect head state
    nullstate.next.append(hmm.getHead())
    hmm.getHead().edges.append((nullstate, 1.0))
    # initialize and connect new non-emitting state
    new_null = CDR.NullState()
    hmm.getTail().next.append(new_null)
    new_null.edges.append((hmm.getTail(), 0.5))
    return new_null


def get_node_ls(sentence, obj_ls):
    """build graph according to given sentence and return the flattened graph

    :param sentence: given sentence object
    :param obj_ls: already initialized hmm models
    :return: flattened graph (node list)
    """
    sentencePath = sentence.name  # get recording path
    # get digit sequence (string)
    _, tail = os.path.split(sentencePath)
    digit_seq = tail.split(".")[0].split("_")[0]
    # initialize starting non-emitting state
    startNull = CDR.NullState()
    # get first silence hmm
    sil_0 = obj_ls[-1]
    # connect starting non-emitting and first silence, get second non-emitting state
    currNull = connect(startNull, sil_0)
    # conncect all digit hmms
    for digit in digit_seq:
        hmm = obj_ls[int(digit)]
        currNull = connect(currNull, hmm)
    # connect end silence
    sil_1 = obj_ls[-2]
    currNull = connect(currNull, sil_1)
    return CDR.flatten(startNull)


def assign_vectors(sentence, path, feature_ls):
    """assign each vector in the sentece to corresponding state (by result of alignment)

    :param sentence: setence object
    :param path: path (result of alignment)
    :param feature_ls: basket list that holds all vectors
    """
    for t in range(len(sentence.val)):
        vector = sentence.val[t]
        id = path[t].id
        # store vector based on state id
        vector_arr = feature_ls[id]
        # vstack all vectors
        vector_arr = np.vstack((vector_arr, vector))
        feature_ls[id] = vector_arr


def update(node_ls, feature_ls):
    """update mean, cov, transition probabilities of all emitting states

    :param node_ls: node list
    :param feature_ls: basket list that holds all vectors
    """
    for node in node_ls:
        # skip null state
        if node.isNull:
            continue
        # get all features belong to this state by state id
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
            # update self transition probability
            if parentNode == node:
                new_edges.append((parentNode, self_trans_prob))
            # update parent transition probability
            else:
                if parentNode.isNull:
                    new_edges.append((parentNode, 1))
                else:
                    new_edges.append((parentNode, next_trans_prob))
        # calculate parent transition probability for child
        next_trans_prob = 1 - self_trans_prob

        node.edges = new_edges
        # update mean
        node.mean = mean
        # update covariance
        node.cov = cov_diag


def train(verbose):
    """
    training digit hmm models from continuous digit sequences
    """
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
        feature_ls = [np.empty((0, 39)) for i in range(73)]  # basket list holds all vectors by state id they belong to
        for sentence in sentence_ls:  # align all sentence in one iteration
            node_ls, nodeNum = get_node_ls(sentence, obj_ls)  # get node list (flattened graph)
            cost, bpt = DTW.DTW(sentence.val, node_ls)  # DTW, alignment
            # print(bpt)
            total_cost += cost
            _, path = DTW.parseBPT(bpt)  # parse back pointer table to get path
            segment = sentence.update_segment(path)
            assign_vectors(sentence, path, feature_ls)  # assign each vector to corresponding state
            """ if verbose:
                # utils.print_seq(path)
                print(segment)
                print("*" * 100) """
        if verbose:
            print("total cost: ", total_cost)
        update(node_ls, feature_ls)  # update all nodes after all alignments complete
        if iteration != 0:
            converge = DTW.check_convergence(prev_cost, total_cost)

        iteration += 1
        prev_cost = total_cost

    # save models
    for obj in obj_ls:
        utils.save_hmm("./model/tz_trained", obj)


start = time()
train(True)
end = time()
print("total wall time: {:.4f}".format(end - start))
