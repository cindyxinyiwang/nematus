'''
Utility functions
'''

import sys
import json
import cPickle as pkl
import numpy
import os
import math
import theano
import theano.tensor as tensor

def align_dot(align, att):
    '''
    Build computational graph for alignment calculation
    align: (batch_size, len1, len2)
    att: (len2, batch_size)
    '''
    scan_func = lambda x, y: tensor.dot(x, y)
    out, update = theano.scan(scan_func,
                    sequences = [align, att.dimshuffle(1, 0)],
                    outputs_info = None,
                    n_steps = align.shape[0])
    return out.reshape((align.shape[1], align.shape[0]))

def get_align_matrix(batch_size, len1, len2, len1_list, len2_list,  align_input, rev=False):
    '''
    len1, len2: length of the padded input 
    len1_list, len2_list: length list of the raw sentence
    '''
    if rev:
        align = numpy.zeros((batch_size, len2, len1))
    else:
        align = numpy.zeros((batch_size, len1, len2))
    for i in range(batch_size):
        align_text = align_input[i]
        toks = align_text.split()
        for tok in toks:
            d = tok.split("-")
            d1, d2 = int(d[0]), int(d[1])
            if rev:
                align[i][d2][d1] = 1.
            else:
                align[i][d1][d2] = 1.
        if rev:
            align[i][len2_list[i]][len1_list[i]] = 1.
        else:
            align[i][len1_list[i]][len2_list[i]] = 1. # align for eos token
    align = align.astype(numpy.float32)
    return align

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def seqs2words(seq, inverse_target_dictionary, join=True):
    words = []
    for w in seq:
        if w == 0:
            break
        if w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            words.append('UNK')
    return ' '.join(words) if join else words

###############
## BLEU calculation

def merge_dict(d1, d2):
    '''
        Merge two dicts. The count of each item is the maximum count in two dicts.
    '''
    result = d1
    for key in d2:
        value = d2[key]
        if result.has_key(key):
            result[key] = max(result[key], value)
        else:
            result[key] = value
    return result
def sentence2dict(sentence, n):
    '''
        Count the number of n-grams in a sentence.

        :type sentence: string
        :param sentence: sentence text

        :type n: int 
        :param n: maximum length of counted n-grams
    '''
    words = sentence.split(' ')
    result = {}
    for n in range(1, n + 1):
        for pos in range(len(words) - n + 1):
            gram = ' '.join(words[pos : pos + n])
            if result.has_key(gram):
                result[gram] += 1
            else:
                result[gram] = 1
    return result

def bleu(hypo_c, refs_c, n):
    '''
        Calculate BLEU score given translation and references.

        :type hypo_c: string
        :param hypo_c: the translations

        :type refs_c: list
        :param refs_c: the list of references

        :type n: int
        :param n: maximum length of counted n-grams
    '''
    correctgram_count = [0] * n
    ngram_count = [0] * n
    hypo_sen = hypo_c.split('\n')
    refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
    hypo_length = 0
    ref_length = 0

    for num in range(len(hypo_sen)):
        hypo = hypo_sen[num]
        h_length = len(hypo.split(' '))
        hypo_length += h_length

        refs = [refs_sen[i][num] for i in range(len(refs_c))]
        ref_lengths = sorted([len(refs[i].split(' ')) for i in range(len(refs))])
        ref_distances = [abs(r - h_length) for r in ref_lengths]

        ref_length += ref_lengths[numpy.argmin(ref_distances)]
        refs_dict = {}
        for i in range(len(refs)):
            ref = refs[i]
            ref_dict = sentence2dict(ref, n)
            refs_dict = merge_dict(refs_dict, ref_dict)

        hypo_dict = sentence2dict(hypo, n)

        for key in hypo_dict:
            value = hypo_dict[key]
            length = len(key.split(' '))
            ngram_count[length - 1] += value
            if refs_dict.has_key(key):
                correctgram_count[length - 1] += min(value, refs_dict[key])

    result = 0.
    bleu_n = [0.] * n
    if correctgram_count[0] == 0:
        return 0.
    for i in range(n):
        if correctgram_count[i] == 0:
            correctgram_count[i] += 1
            ngram_count[i] += 1
        bleu_n[i] = correctgram_count[i] * 1. / ngram_count[i]
        result += math.log(bleu_n[i]) / n
    bp = 1
    if hypo_length < ref_length:
        bp = math.exp(1 - ref_length * 1.0 / hypo_length)
    return bp * math.exp(result)

def bleu_file(hypo, refs):
    '''
        Calculate the BLEU score given translation files and reference files.

        :type hypo: string
        :param hypo: the path to translation file

        :type refs: list
        :param refs: the list of path to reference files
    '''
    hypo = open(hypo, 'r').read()
    refs = [open(ref, 'r').read() for ref in refs]
    return bleu(hypo, refs, 4)


def get_ref_files(ref):
    '''
        Get the list of reference files by prefix.
        Suppose nist02.en0, nist02.en1, nist02.en2, nist02.en3 are references and nist02.en does not exist,
        then get_ref_files("nist02.en") = ["nist02.en0", "nist02.en1", "nist02.en2", "nist02.en3"]

        :type ref: string
        :param ref: the prefix of reference files
    '''
    if os.path.exists(ref):
        return [ref]
    else:
        ref_num = 0
        result = []
        while os.path.exists(ref + str(ref_num)):
            result.append(ref + str(ref_num))
            ref_num += 1
        return result
