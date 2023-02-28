import pickle
import queue
import numpy as np
from num2words import num2words
import os


class HMM:
    """
    HMM object defined in project 3
    """

    def __init__(self, state_ls, trans_mat, name):
        self.state_ls = state_ls
        self.trans_mat = trans_mat
        self.name = name


def save_hmm(folder, hmm):
    filename = os.path.join(folder, hmm.name + ".pickle")
    with open(filename, "wb") as file:
        pickle.dump(hmm, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_hmm(folder, digit):
    for file in os.listdir(folder):
        if digit in file:
            filename = os.path.join(folder, file)
            break
    with open(filename, "rb") as file:
        return pickle.load(file)


def dfs_print(head):
    # depth first print tree
    if head == None:
        return
    head.visited = True
    print(head)
    for child in head.next:
        if not child.visited:
            dfs_print(child)


def countNode(head):
    if head == None:
        return 0
    else:
        count = 0
        for child in head.next:
            if child.visited:
                continue
            child.visited = True
            count += countNode(child)
        return 1 + count


def parseSName(filename):
    """
    Get correct answer from file name
    """
    sentence = filename.split(".")[0]
    answer = []
    for char in sentence:
        answer.append(num2words(char))
    return answer


def getDis(d1, d2):
    if d1 == "*" or d2 == "*":
        return 0
    if d1 == d2:
        return 0
    else:
        return 1


def dtw(template, test):
    """Preform dtw on two list to calculate the word error rate
    :param template: correct answer
    :param test: recognized result
    :return : minimum edit distance between two list
    """
    template = ["*"] + template
    trellis = [i for i in range(len(template))]
    for digit in test:
        prev = trellis.copy()
        for i in range(len(trellis)):
            if i == 0:
                trellis[i] = prev[i] + 1
            else:
                candidates = [prev[i] + 1, trellis[i - 1] + 1, prev[i - 1] + getDis(template[i], digit)]
                trellis[i] = np.min(candidates)
    return trellis[-1]


def uniform_seq(length, node_ls):
    num = len(node_ls) - 2
    tmp_seg_len = length // num
    resil = length % num
    idx = 0
    segment = []
    for i in range(num):
        seg_len = tmp_seg_len
        if resil > 0:
            seg_len = tmp_seg_len + 1
            resil -= 1
        segment.append((idx, idx + seg_len))
        idx += seg_len
    seq = []
    for i in range(num):
        seq = seq + [node_ls[i + 1]] * (segment[i][1] - segment[i][0])
    return segment, seq


def print_seq(seq):
    names = []
    for node in seq:
        names.append(node.name)
    print(names)
