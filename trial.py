import Test
import utils
import os



def test(filename, lpc, whethershow):
    """Recognize a singel sentence and calculate accuracies
    :param filename: tar sentence name
    :param verbose: if to print results
    :param force7digit: if to force the model to have 7 digits
    :return: correct digit number, total digit number, minimum edit distance, sentence correct rate
    """
    #  correct answer from file name
    answer = utils.parseSName(filename)
    p1_folder = "./sequences/tz"
    filepath = os.path.join(p1_folder, filename)
    # calculate mfcc features
    result = Test.test(filepath, lpc)
    total = len(answer) #  total digit number
    count = 0
    # results and answers may have different length
    cor_results = []
    for results in result:
        new = ''.join(results.split('0'))
        cor_results.append(new)
    result = cor_results    
    for i in range(total):
        try:
            if answer[i] == result[i]:
                count += 1
        except:
            break
    minEditDis = utils.dtw(answer, result)
    wre = minEditDis / total
    if whethershow:
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

def testMany(testSet, lpc, whethershow):
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
        count, total, minEditDis, sentenceCorrect= test(file, lpc, whethershow)
        totalNum += total
        correctNum += count
        medSum += minEditDis
        s_correctNum += sentenceCorrect
    print("Sentence correct rate: {:.2f}".format(s_correctNum/len(testSet)) )
    print("Digit correct rate: {:.2f}".format(correctNum/totalNum))
    print("Word error rate: {:.2f}".format(medSum/totalNum))
    return medSum/totalNum
    
