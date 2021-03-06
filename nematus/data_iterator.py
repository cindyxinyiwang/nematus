import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 multi_src=False):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                
                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()


        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss = tmp

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        return source, target

class MultiSrcTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 align1_file=None,
                 align2_file=None):
        self.source_files = source.split(",")
        if shuffle_each_epoch:
            self.source_orig1 = self.source_files[0]
            self.source_orig2 = self.source_files[1]
            self.target_orig = target
            if align1_file:
                self.align_orig1 = align1_file
                self.align_orig2 = align2_file
                self.source1, self.source2, self.target, self.align1, self.align2 = shuffle.main([self.source_orig1, self.source_orig2, self.target_orig, self.align_orig1, self.align_orig2], temporary=True)
            else:
                self.source1, self.source2, self.target = shuffle.main([self.source_orig1, self.source_orig2, self.target_orig], temporary=True)
                self.align1 = None
        else:
            self.source1 = fopen(self.source_files[0], 'r') 
            self.source2 = fopen(self.source_files[1], 'r') 
            self.target = fopen(target, 'r')
            if align1_file:
                self.align1 = fopen(align1_file, 'r')
                self.align2 = fopen(align2_file, 'r')
            else:
                self.align1 = None
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source1_buffer = []
        self.source2_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        if self.align1:
            self.align1_buffer = []
            self.align2_buffer = []

        '''
        ts = []
        for s in self.source1:
            ts.append(s)
        print len(ts)
        self.source1.seek(0)

        ts = []
        for s in self.source2:
            ts.append(s)
        print len(ts)
        self.source2.seek(0)

        ts = []
        tss = []
        for s, ss in zip(self.source1, self.source2):
            ts.append(s)
            tss.append(ss)
        print len(ts), len(tss)
        self.source1.seek(0)
        self.source2.seek(0)      
        '''

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            if self.align1:
                self.source1, self.source2, self.target, self.align1, self.align2 = shuffle.main([self.source_orig1, self.source_orig2, self.target_orig, self.align_orig1, self.align_orig2], temporary=True)
            else:
                self.source1, self.source2, self.target = shuffle.main([self.source_orig1, self.source_orig2, self.target_orig], temporary=True)
        else:
            self.source1.seek(0)
            self.source2.seek(0)
            self.target.seek(0)
            if self.align1:
                self.align1.seek(0)
                self.align2.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source1 = []
        source2 = []
        target = []
        if self.align1:
            align1 = []
            align2 = []

        #print len(self.source1_buffer), len(self.source2_buffer), len(self.target_buffer)
        # fill buffer, if it's empty
        assert len(self.source1_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        assert len(self.source2_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        if self.align1:
            assert len(self.align1_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
            assert len(self.align2_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source1_buffer) == 0:
            for ss1 in self.source1:
                ss1 = ss1.split()
                ss2 = self.source2.readline().split()
                tt = self.target.readline().split()
                if self.align1:
                    a1 = self.align1.readline()
                    a2 = self.align2.readline()

                if self.skip_empty and (len(ss1) == 0 or len(ss2) == 0 or len(tt) == 0):
                    continue
                if len(ss1) > self.maxlen or len(ss2) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.source1_buffer.append(ss1)
                self.source2_buffer.append(ss2)
                self.target_buffer.append(tt)
                if self.align1:
                    self.align1_buffer.append(a1)
                    self.align2_buffer.append(a2)
                if len(self.source1_buffer) == self.k:
                    break

            if len(self.source1_buffer) == 0 or len(self.source2_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf1 = [self.source1_buffer[i] for i in tidx]
                _sbuf2 = [self.source2_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                if self.align1:
                    _sa1 = [self.align1_buffer[i] for i in tidx]
                    _sa2 = [self.align2_buffer[i] for i in tidx]
                    self.align1_buffer = _sa1
                    self.align2_buffer = _sa2

                self.source1_buffer = _sbuf1
                self.source2_buffer = _sbuf2
                self.target_buffer = _tbuf

            else:
                self.source1_buffer.reverse()
                self.source2_buffer.reverse()
                self.target_buffer.reverse()
                if self.align1:
                    self.align1_buffer.reverse()
                    self.align2_buffer.reverse()

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss1 = self.source1_buffer.pop()
                    ss2 = self.source2_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss1:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss1 = tmp

                tmp = []
                for w in ss2:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[1][w] if w in self.source_dicts[1] else 1]
                    tmp.append(w)
                ss2 = tmp

                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                source1.append(ss1)
                source2.append(ss2)
                target.append(tt)

                if self.align1:
                    align1.append(self.align1_buffer.pop())
                    align2.append(self.align2_buffer.pop())

                if len(source1) >= self.batch_size or \
                        len(target) >= self.batch_size or \
                        len(source2) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
        if self.align1:
            return [source1, source2], target, align1, align2
        else:
            return [source1, source2], target, None, None
