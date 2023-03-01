import mfcc


class Sentence:
    def __init__(self, name):
        self.name = name
        self.val = None
        self.seq = None
        self.segment = None
        vectors = mfcc.mfcc_features(name, 40)
        self.val = vectors

    def update_segment(self, seq):
        # utils.print_seq(seq)
        self.seq = seq
        segment = []
        start_idx = 0
        curr_idx = 0
        curr_state = self.seq[0]
        for i in range(len(self.seq)):
            state = self.seq[i]
            if state != curr_state:
                segment.append((start_idx, i))
                start_idx = i
                curr_state = state
            if i == len(self.seq) - 1:
                segment.append((start_idx, i + 1))
        self.segment = segment
        return segment

    def get_vectors(self, id):
        start_idx, end_idx = self.segment[id]
        return self.val[start_idx:end_idx]

    def get_vectorNum(self, id):
        return self.segment[id][1] - self.segment[id][0]
