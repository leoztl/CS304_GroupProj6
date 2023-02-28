import os
import re
import CDR
from num2words import num2words


folder_path = "./p1_sentence"

# read files from a folder and return a list of filenames without '.wav' extension
def read_files(path):

    file_names = []

    for filename in os.listdir(path):
        match = re.match(r'(.*)\.wav', filename)
        if match:
            file_name_without_extension = match.group(1)
            file_names.append(file_name_without_extension)

    return file_names

# transform a digit-string to a node list that connects each other
def load_initialized(file_name):
    
    startNull = CDR.NullState()
    startHmm = CDR.Hmm('sil', 0)
    currentNull = CDR.NullState()
    connect(startNull, startHmm, currentNull)
    
    
    for i in range(len(file_name)):
        number = file_name[i]
        idx = i+1
        
        digitHmm = CDR.Hmm(num2words(number), i)
        nextNull = CDR.NullState()
        connect(currentNull, digitHmm, nextNull)
        currentNull = nextNull
    
    endNull = CDR.NullState()
    endHmm = CDR.Hmm('sil', idx+1)
    connect(currentNull, endHmm, endNull)
    
    return CDR.flatten(startNull)
      
# connect hmm with its starting and ending null node
def connect(null1, hmm, null2):
    null1.next.append(hmm.getHead())
    hmm.getHead().edges.append((null1, 1))

    hmm.getTail().next.append(null2)
    null2.edges.append((hmm.getTail(), 1))
