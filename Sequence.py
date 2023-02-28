from CDR import *
import argparse
import os
from num2words import num2words


def build_spec(l):
    """
    Build a graph of 4-digit or 7-digit telephone numbers
    """
    '''
    word_ls = []
    startNull = NullState()
    startWord = Word([num2words(i) for i in range(10)], 0)
    word_ls.append(startWord)
    currentNull1 = appendWord(startNull, startWord)
    currentWord1 = Word([num2words(i) for i in range(10)], 0)
    word_ls.append(currentWord1)
    currentNull2 = appendWord(currentNull1, currentWord1)
    currentNull1.edges.append((currentNull2, 0.5))
    '''
    word_ls = []
    StartNull = NullState()
    currentNull = StartNull
    for letter in l:
        currentWord = Word(num2words(letter) , 0)
        word_ls.append(currentWord)
        currentNull = appendWord(currentNull, currentWord)
    return StartNull, currentWord


def RSS(sentence, node_ls, nodeNum, currentWord, loopbackcost):
    """Recognize single sentence

    :param sentence: mfcc features of recording
    :param node_ls: flattened graph, a list of state, including non-emitting state
    :param nodeNum: total state number, including non-emitting state
    :param branchNull: branchNull to allow skip of area code
    :param force7digit: if to force the model to be 7-digit
    :return: recognition results
    """
    startNull = node_ls[0]
    startNull.currDis = 0
    print(currentWord)
    tails = currentWord.getAllTails()
    tailBP = []
    '''
    if not force7digit:
        branchNull.currDis = 0
    '''
    bpt = [[None for i in range(nodeNum)] for j in range(len(sentence))]
    # print(len(bpt))
    for t in range(len(sentence)):
        vector = sentence[t]
        for currentNode in node_ls:
            currentNode.prevDis = currentNode.currDis
            if currentNode == startNull:
                minTail = tails[0]
                for n in tails:
                    if minTail.currDis > n.currDis:
                        minTail = n
                currentNode.currDis = minTail.currDis + loopbackcost
                tailBP.append(minTail)
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
                minParent = tailBP[t-1]
            distance = parentDis[minIdx]
            bpt[t][currentNode.id] = minParent
            if currentNode.isNull:
                currentNode.currDis = distance
            else:
                currentNode.currDis = distance + currentNode.getDis(vector)
    return parseBPT(bpt)

def main(args):
    sentence = args.ss
    p1_folder = "./p1_sentence"
    if sentence != None:
        answer = utils.parseSName(sentence)
        filename = os.path.join(p1_folder, sentence)
        sentence = mfcc.mfcc_features(filename, 40)
        startNull, branchNull = buildall()
        node_ls, nodeNum = flatten(startNull)
        result, seq = RSS(sentence, node_ls, nodeNum, branchNull, loopbackcost = 10000)
        #result, seq = RSS(sentence, node_ls, nodeNum, branchNull, False)
        total = len(answer)
        count = 0
        for i in range(total):
            try:
                if answer[i] == result[i]:
                    count += 1
            except:
                break
        print("Result: ", result)
        print("Correct rate: {:.2f}".format(count / total))
        minEditDis = utils.dtw(answer, result)
        print("Minimum edit distance: {}\nWord error rate: {:.2f}".format(minEditDis, minEditDis / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="tz")
    parser.add_argument("--ss", default=None)
    args = parser.parse_args()
    main(args)

def test(filename,verbose, force7digit):
    """Recognize a singel sentence and calculate accuracies
    :param filename: tar sentence name
    :param verbose: if to print results
    :param force7digit: if to force the model to have 7 digits
    :return: correct digit number, total digit number, minimum edit distance, sentence correct rate
    """
    #  correct answer from file name
    answer = utils.parseSName(filename)
    p1_folder = "./p1_sentence"
    filepath = os.path.join(p1_folder, filename)
    # calculate mfcc features
    sentence = mfcc.mfcc_features(filepath, 40)
    # build graph
    startNull, currentWord = buildall()
    # flatten graph
    node_ls, nodeNum = flatten(startNull)
    # recognize sentence
    result, seq = RSS(sentence, node_ls, nodeNum, currentWord, loopbackcost = 300)
    total = len(answer) #  total digit number
    count = 0
    # results and answers may have different length
    for i in range(total):
        try:
            if answer[i] == result[i]:
                count += 1
        except:
            break
    minEditDis = utils.dtw(answer, result)
    wre = minEditDis / total
    if verbose:
        print("Correct answer: ", answer)
        print("Result: ", result)
        print("Correct rate: {:.2f}".format(count / total))
        print("Minimum edit distance: {}\nWord error rate: {:.2f}".format(minEditDis, wre))
        print("*"*50)
    if minEditDis == 0:
        sentenceCorrect = 1
    else:
        sentenceCorrect = 0
    return count, total, minEditDis, sentenceCorrect

def testMany(testSet, verbose, force7digit=False):
    """Recognize a set of sentences one by one and print the results
    :param testSet: sentence name list, digit4 or digit7 or all
    :param force7digit: if to force the model to have 7 digits
    :param verbose: if to print the results for each sentence during testing
    """
    totalNum = 0
    correctNum = 0
    medSum = 0
    s_correctNum = 0
    for file in testSet:
        count, total, minEditDis, sentenceCorrect= test(file, verbose, force7digit)
        totalNum += total
        correctNum += count
        medSum += minEditDis
        s_correctNum += sentenceCorrect
    print("Sentence correct rate: {:.2f}".format(s_correctNum/len(all)) )
    print("Digit correct rate: {:.2f}".format(correctNum/totalNum))
    print("Word error rate: {:.2f}".format(medSum/totalNum))
    
folderPath = "./p1_sentence"
digit4 = []
digit7 = []
for file in os.listdir(folderPath):
    name = file.split(".")[0]
    if len(name) == 4:
        digit4.append(file)
    elif len(name) == 7:
        digit7.append(file)
    else:
        print("Unexpected sentence length!")
all = digit4 + digit7
testMany(all, True)