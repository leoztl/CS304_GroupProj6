import DTW
import os
import numpy as np
import sentence as st
import utils
import CDR
import argparse


def update(node_ls, sentence_ls):
    """update transition probabilities, mean, covariance for all emitting node

    :param node_ls: node list
    :param sentence_ls: sentence list 
    """
    state_idx = 0
    for node in node_ls:
        if node.isNull:
            continue
        vectorNum = 0
        vector_arr = np.empty((0, 39))
        for sentence in sentence_ls:
            stateNum += sentence.get_vectorNum(state_idx)
            state_vectors = sentence.get_vectors(state_idx)
            vector_arr = np.vstack((vector_arr, state_vectors))
        # calculate self transition probability
        self_trans_prob = (vectorNum - len(sentence_ls)) / vectorNum
        next_trans_prob = 1 - self_trans_prob
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
                if state_idx == 0:
                    new_edges.append((parentNode, 1))
                else:
                    new_edges.append((parentNode, next_trans_prob))

        node.edges = new_edges
        # update mean
        node.mean = mean
        # update covariance
        node.cov = cov_diag
        # move to next emitting state
        state_idx += 1


def initialize(wavefolder, digit):
    sentence_ls = []
    sentence_num = 0
    stateNum = 5
    node_ls = initialize_node_ls(digit)
    for file in os.listdir(wavefolder):
        if digit in file:
            filename = os.path.join(wavefolder, file)
            sentence = st.Sentence(filename)
            _, uni_seq = utils.uniform_seq(len(sentence.val), node_ls)
            sentence.update_segment(uni_seq)
            sentence_ls.append(sentence)
            sentence_num += 1
    update(node_ls, sentence_ls)
    return node_ls, sentence_ls


def initialize_node_ls(digit, stateNum=5):
    n0 = CDR.Node(None, None, digit + "0")
    n1 = CDR.Node(None, None, digit + "1")
    n2 = CDR.Node(None, None, digit + "2")
    n3 = CDR.Node(None, None, digit + "3")
    n4 = CDR.Node(None, None, digit + "4")
    n0.edges.append((n0, 0.5))
    n1.edges.append((n0, 0.5))
    n1.edges.append((n1, 0.5))
    n2.edges.append((n1, 0.5))
    n2.edges.append((n2, 0.5))
    n3.edges.append((n2, 0.5))
    n3.edges.append((n3, 0.5))
    n4.edges.append((n3, 0.5))
    n4.edges.append((n4, 1))
    startNull = CDR.NullState()
    endNull = CDR.NullState()
    n0.edges.append((startNull, 1))
    startNull.next.append(n0)
    endNull.edges.append((n4, 0.5))
    node_ls = [startNull, n0, n1, n2, n3, n4, endNull]
    for i in range(len(node_ls)):
        node_ls[i].id = i
    return node_ls


def train(digit, verbose):
    wave_folder = "./data/tzwave/all"
    node_ls, sentence_ls = initialize(wave_folder, digit)
    converge = False
    iteration = 0
    prev_cost = None
    while not converge:
        if verbose:
            print("iteration: ", iteration)
        total_cost = 0
        for sentence in sentence_ls:
            cost, bpt = DTW.DTW(sentence.val, node_ls)
            total_cost += cost
            _, path = DTW.parseBPT(bpt)
            segment = sentence.update_segment(path)
            if verbose:
                print(segment)
        update(node_ls, sentence_ls)
        if iteration != 0:
            converge = DTW.check_convergence(prev_cost, total_cost)

        iteration += 1
        prev_cost = total_cost
    state_ls = []
    trans_mat = np.empty((0, 5))
    for i in range(len(node_ls)):
        node = node_ls[i]
        if node.isNull:
            continue
        state_tuple = (node.mean, node.cov)
        trans_prob = np.zeros(5)
        for edge in node.edges:
            if edge[0] == node:
                self_trans = edge[1]
        trans_prob[i - 1] = self_trans
        if i != len(node_ls) - 2:
            trans_prob[i] = 1 - self_trans

        state_ls.append(state_tuple)
        trans_mat = np.vstack((trans_mat, trans_prob))
    trans_mat = np.where(trans_mat == 0, np.finfo(np.float64).eps, trans_mat)
    hmm = utils.HMM(state_ls, trans_mat, digit)
    utils.save_hmm("./model/tz", hmm)
    return node_ls


def main(args):
    digit = args.word
    verbose = args.verbose
    train(digit, verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", "-w", default="zero")
    parser.add_argument("--verbose", "-v", default="False")
    args = parser.parse_args()
    main(args)
