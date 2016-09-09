import theano
import theano.tensor as T
import numpy as np
import sys
import codecs
from collections import Counter
import math
import argparse

sys.path.append('/u/subramas/Research/SanDeepLearn/')

import recurrent
import layer
import optimizers
from recurrent import LSTM, FastLSTM, MiLSTM, BiRNN, GRU
from layer import FullyConnectedLayer, EmbeddingLayer
from optimizers import Optimizer

theano.config.floatX='float32'

parser = argparse.ArgumentParser()
parser.add_argument(
        "-train_src",
        "--train_src_sentences",
        help="path to train src sentences",
        required=True
    )
parser.add_argument(
        "-train_tgt",
        "--train_tgt_sentences",
        help="path to train tgt sentences",
        required=True
    )
parser.add_argument(
        "-dev_src",
        "--dev_src_sentences",
        help="path to dev src sentences",
        required=True
    )
parser.add_argument(
        "-dev_tgt",
        "--dev_tgt_sentences",
        help="path to dev tgt sentences",
        required=True
    )

parser.add_argument(
        "-batch_size",
        "--batch_size",
        help="batch size",
        required=True
    )

args = parser.parse_args()

data_path_train_src = args.train_src_sentences
data_path_train_tgt = args.train_tgt_sentences
data_path_dev_src = args.dev_src_sentences
data_path_dev_tgt = args.dev_tgt_sentences
batch_size = int(args.batch_size)

def partition_data(src_lines, tgt_lines):
    train_src = src_lines[:int(0.8 * len(src_lines))]
    train_tgt = tgt_lines[:int(0.8 * len(tgt_lines))]
    dev_src = src_lines[int(0.8 * len(src_lines)): int(0.9 * len(src_lines))]
    dev_tgt = tgt_lines[int(0.8 * len(tgt_lines)): int(0.9 * len(tgt_lines))]
    test_src = src_lines[int(0.9 * len(src_lines)):]
    test_tgt = tgt_lines[int(0.9 * len(tgt_lines)):]
    return train_src, train_tgt, dev_src, dev_tgt, test_src, test_tgt

def get_vocab(lines):
    vocab = set()
    for line in lines:
        for word in line:
            vocab.add(word)
    vocab.add('<s>')
    vocab.add('</s>')
    vocab.add('<NULL>')
    word2ind = {}
    ind2word = {}

    for ind, word in enumerate(vocab):
        word2ind[word] = ind
        ind2word[ind] = word
    return vocab, word2ind, ind2word

def prepare_batch(src_sentences, tgt_sentences):
    src_sentences = [['<s>'] + sent + ['</s>'] for sent in src_sentences]
    tgt_sentences = [['<s>'] + sent + ['</s>'] for sent in tgt_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    tgt_lens = [len(sent) for sent in tgt_sentences]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    src_sentences = [[src_word2ind[word] for word in sent] + [src_word2ind['<NULL>']] * (max_src_len - len(sent)) for sent in src_sentences]
    tgt_sentences_inp = [[tgt_word2ind[word] for word in sent[:-1]] + [tgt_word2ind['<NULL>']] * (max_tgt_len - len(sent)) for sent in tgt_sentences]
    tgt_sentences_op = [[tgt_word2ind[word] for word in sent[1:]] + [tgt_word2ind['<NULL>']] * (max_tgt_len - len(sent)) for sent in tgt_sentences]
    tgt_mask = np.zeros((len(tgt_sentences), len(tgt_sentences_op[0]))).astype(np.float32)
    for ind, tgt_len in enumerate(tgt_lens):
        tgt_mask[ind][:tgt_len - 1] = 1.
    src_sentences = np.array(src_sentences).astype(np.int32)
    tgt_sentences_inp = np.array(tgt_sentences_inp).astype(np.int32)
    tgt_sentences_op = np.array(tgt_sentences_op).astype(np.int32)
    src_lens = np.array(src_lens).astype(np.int32)
    return src_sentences, tgt_sentences_inp, tgt_sentences_op, src_lens, tgt_mask

def prepare_evaluation_batch(src_sentences):
    src_sentences = [['<s>'] + sent + ['</s>'] for sent in src_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    max_src_len = max(src_lens)
    src_sentences = [[src_word2ind[word] for word in sent] + [src_word2ind['<NULL>']] * (max_src_len - len(sent)) for sent in src_sentences]
    src_sentences = np.array(src_sentences).astype(np.int32)
    return src_sentences, src_lens

def decode_batch(src_sentences):
    src_sentences, src_lens = prepare_evaluation_batch(src_sentences)
    decode_state = np.array([[tgt_word2ind['<s>']] for sent in batch_tgt_inp]).astype(np.int32)
    is_decoding = [True] * len(src_sentences)
    decode_length = 1
    while any(is_decoding) and decode_length < 25:
        next_words = f_eval(
            src_sentences,
            decode_state,
            src_lens
        )
        next_words = [x[0] for x in np.argmax(next_words, axis=2)]
        is_finished = [word == tgt_word2ind['</s>'] for word in next_words]
        for ind, item in enumerate(is_finished):
            if item == True:
                is_decoding[ind] = False
        decode_state = np.c_[decode_state, next_words].astype(np.int32)
        decode_length += 1
    return decode_state

def decode_dev():
    decoded_sentences = []
    for i in range(0, len(dev_src), batch_size):
        print 'Decoding batch %d out of %d ' % (i, len(dev_src))
        decoded_batch = decode_batch(dev_src[i:i + batch_size])
        decoded_sentences += decoded_batch
    return decoded_sentences

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1,5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))
    return stats

def bleu(stats):
    if len(filter(lambda x: x==0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)

def get_validation_bleu(hypotheses, reference):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return "%.3f" % (100*bleu(stats))

train_src = [line.strip().split() for line in codecs.open(data_path_train_src, 'r', encoding='utf-8')]
train_tgt = [line.strip().split() for line in codecs.open(data_path_train_tgt, 'r', encoding='utf-8')]

dev_src = [line.strip().split() for line in codecs.open(data_path_dev_src, 'r', encoding='utf-8')]
dev_tgt = [line.strip().split() for line in codecs.open(data_path_dev_tgt, 'r', encoding='utf-8')]

src_vocab, src_word2ind, src_ind2word = get_vocab(train_src)
tgt_vocab, tgt_word2ind, tgt_ind2word = get_vocab(train_tgt)

print '=========================================================='
print 'Finished reading data ...'
print '=========================================================='
print 'Number of training sentence-pairs : %d ' % (len(train_src))
print 'Number of validation sentence-pairs : %d ' % (len(dev_src))
print '=========================================================='
print 'Source vocabulary size : %d ' % (len(src_vocab))
print 'Target vocabulary size : %d ' % (len(src_vocab))
print '=========================================================='

src_inp = T.imatrix()
tgt_inp = T.imatrix()
tgt_op = T.imatrix()
src_lens = T.ivector()
tgt_mask = T.fmatrix()
index = T.scalar()

src_inp_t = np.random.randint(low=1, high=100, size=(5, 10)).astype(np.int32)
tgt_inp_t = np.random.randint(low=1, high=100, size=(5, 15)).astype(np.int32)
tgt_op_t = np.random.randint(low=1, high=200, size=(5, 15)).astype(np.int32)
src_lens_t = np.random.randint(low=1, high=9, size=(5,)).astype(np.int32)
tgt_mask_t = np.float32(np.random.rand(5, 15).astype(np.float32) > 0.5)

#Model
src_emb_dim      = 256  # source word embedding dimension
tgt_emb_dim      = 256  # target word embedding dimension
src_lstm_op_dim  = 512  # source LSTMs hidden dimension
tgt_lstm_op_dim  = 2 * src_lstm_op_dim  # target LSTM hidden dimension
beta = 500 # Regularization coefficient

n_src = len(src_word2ind)  # number of words in the source language
n_tgt = len(tgt_word2ind)  # number of words in the target language

# Embedding Lookup Tables
src_embedding_layer = EmbeddingLayer(
    input_dim=n_src,
    output_dim=src_emb_dim,
    name='src_embedding'
)
tgt_embedding_layer = EmbeddingLayer(
    input_dim=n_tgt,
    output_dim=tgt_emb_dim,
    name='tgt_embedding'
)

# Encoder BiLSTM and Decoder LSTM
src_lstm_forward = FastLSTM(
    input_dim=src_emb_dim,
    output_dim=src_lstm_op_dim,
    name='src_lstm_forward'
    
)
src_lstm_backward = FastLSTM(
    input_dim=tgt_emb_dim,
    output_dim=src_lstm_op_dim,
    name='src_lstm_backward'
)
tgt_lstm = FastLSTM(
    input_dim=tgt_emb_dim,
    output_dim=tgt_lstm_op_dim,
    name='tgt_lstm'
)

# Projection layers
tgt_lstm_h_to_vocab = FullyConnectedLayer(
    input_dim=tgt_lstm_op_dim + 2 * src_lstm_op_dim,
    output_dim=n_tgt,
    batch_normalization=False,
    activation='softmax',
)

params = src_embedding_layer.params + tgt_embedding_layer.params +     src_lstm_forward.params + src_lstm_backward.params +     tgt_lstm.params[:-1] + tgt_lstm_h_to_vocab.params # No need to learn h_0 for target LSTM

print 'Model parameters ...'
print '=========================================================='
print 'Src Embedding dim : %d ' % (src_emb_dim)
print 'Tgt Embedding dim : %d ' % (tgt_emb_dim)
print 'Encoder BiLSTM dim : %d ' % (2 * src_lstm_forward.output_dim)
print 'Decoder LSTM dim : %d ' %(tgt_lstm.output_dim)
print '=========================================================='

#Get embedding matrices
src_emb_inp = src_embedding_layer.fprop(src_inp)
tgt_emb_inp = tgt_embedding_layer.fprop(tgt_inp)

# Get BiLSTM representations
src_lstm_forward.fprop(src_emb_inp)
src_lstm_backward.fprop(src_emb_inp[::-1, :])
encoder_representation = T.concatenate((src_lstm_forward.h, src_lstm_backward.h[::-1, :]), axis=2)
encoder_final_state = encoder_representation.dimshuffle(1, 0, 2)[T.arange(src_inp.shape[0]), src_lens - 1, :]

# Get Target LSTM representation & Attention Vectors
tgt_lstm.h_0 = encoder_final_state
tgt_lstm.fprop(tgt_emb_inp)

# Attention
attention = T.batched_dot(tgt_lstm.h.dimshuffle(1, 0, 2), encoder_representation.dimshuffle(1, 2, 0))
attention = T.batched_dot(attention, encoder_representation.dimshuffle(1, 0, 2))

# Concatenate the attention vectors to the Target LSTM output before predicting the next word
target_representation = T.concatenate([attention, tgt_lstm.h.dimshuffle(1, 0, 2)], axis=2)

# Project to target vocabulary and softmax
proj_layer_input = target_representation.reshape((tgt_inp.shape[0] * tgt_inp.shape[1], target_representation.shape[2]))
proj_output_rep = tgt_lstm_h_to_vocab.fprop(proj_layer_input)
final_output = proj_output_rep.reshape((tgt_inp.shape[0], tgt_inp.shape[1], len(tgt_vocab)))
final_output = T.clip(final_output, 1e-5, 1-1e-5)

# Compute cost
cost = - (T.log(final_output[
    T.arange(tgt_emb_inp.shape[0]).dimshuffle(0, 'x').repeat(tgt_emb_inp.shape[1], axis=1).flatten(),
    T.arange(tgt_emb_inp.shape[1]).dimshuffle('x', 0).repeat(tgt_emb_inp.shape[0], axis=0).flatten(),
    tgt_op.flatten()
]) * tgt_mask.flatten()).sum() / T.neq(tgt_mask, 0).sum()

cost += beta * T.mean((tgt_lstm.h[:,:-1] ** 2 - tgt_lstm.h[:,1:] ** 2) ** 2) # Regularization of RNNs from http://arxiv.org/pdf/1511.08400v6.pdf

print 'Computation Graph Node Shapes ...'
print '=========================================================='
print 'src embedding dim', src_emb_inp.eval({src_inp:src_inp_t}).shape
print 'tgt embedding dim', tgt_emb_inp.eval({tgt_inp:tgt_inp_t}).shape
print 'encoder forward dim', src_lstm_forward.h.dimshuffle(1, 0, 2).eval({src_inp:src_inp_t}).shape
print 'encoder backward dim', src_lstm_backward.h.dimshuffle(1, 0, 2).eval({src_inp:src_inp_t}).shape
print 'encoder hidden state shape', encoder_representation.eval({src_inp:src_inp_t}).shape
print 'encoder final state shape', encoder_final_state.eval({src_inp:src_inp_t, src_lens:src_lens_t}).shape
print 'decoder hidden state shape', tgt_lstm.h.dimshuffle(1, 0, 2).eval({src_inp: src_inp_t, tgt_inp:tgt_inp_t, src_lens:src_lens_t}).shape
print 'attention vectors shape', attention.eval({tgt_inp:tgt_inp_t, src_inp:src_inp_t, src_lens:src_lens_t}).shape
print 'decoder + attention shape', target_representation.eval({tgt_inp:tgt_inp_t, src_inp:src_inp_t, src_lens:src_lens_t}).shape
print 'tgt vocab projection shape', proj_output_rep.eval({tgt_inp:tgt_inp_t, src_inp:src_inp_t, src_lens:src_lens_t}).shape
print 'tgt vocab softmax shape', final_output.eval({tgt_inp:tgt_inp_t, src_inp:src_inp_t, src_lens:src_lens_t}).shape
print 'cost', cost.eval({tgt_inp:tgt_inp_t, src_inp:src_inp_t, tgt_op:tgt_op_t, src_lens:src_lens_t, tgt_mask:tgt_mask_t})
print '=========================================================='

print 'Compiling theano functions ...'
print '=========================================================='

updates=Optimizer(clip=5.0).adam(
    cost=cost,
    params=params
)

f_train = theano.function(
    inputs=[src_inp, tgt_inp, tgt_op, src_lens, tgt_mask],
    outputs=cost,
    updates=updates
)

f_eval = theano.function(
    inputs=[src_inp, tgt_inp, src_lens],
    outputs=final_output,
)

num_epochs = 100
print 'Training network ...'
print '=========================================================='
for i in range(num_epochs):
    dev_predictions = decode_dev()
    print 'Epoch : %d dev BLEU : %.3f' % (i, get_validation_bleu(dev_predictions, dev_tgt))
    print '=========================================================='
    for j in xrange(0, len(train_src), batch_size):
        batch_src, batch_tgt_inp, batch_tgt_op, batch_src_lens, batch_tgt_mask = prepare_batch(train_src[j:j + batch_size], train_tgt[j:j + batch_size])
        entropy = f_train(
            batch_src,
            batch_tgt_inp,
            batch_tgt_op,
            batch_src_lens,
            batch_tgt_mask
        )
        print 'Epoch : %d Minibatch : %d Loss : %.3f' % (i, j, entropy)
