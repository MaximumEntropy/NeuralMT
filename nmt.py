"""Neural Machine Translation."""
import theano
import theano.tensor as T
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs
from collections import Counter
import math
import argparse
import logging
from sklearn.utils import shuffle
import os
import pickle

sys.path.append('/u/subramas/Research/SanDeepLearn/')

from recurrent import FastLSTM, GRU
from layer import FullyConnectedLayer, EmbeddingLayer
from optimizers import Optimizer
from config import src_emb_dim, tgt_emb_dim, src_lstm_op_dim, tgt_lstm_op_dim, \
    beta

theano.config.floatX = 'float32'

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
parser.add_argument(
    "-exp",
    "--experiment_name",
    help="name of the experiment",
    required=True
)
parser.add_argument(
    "-seed",
    "--seed",
    help="seed for pseudo random number generator",
    default=1337
)
parser.add_argument(
    "-att",
    "--attention",
    help="attention mechanism - dot, mlp etc.",
    default='dot'
)
parser.add_argument(
    "-pre_src",
    "--pretrained_src",
    help="path to src pretrained vectors",
    default='none'
)
parser.add_argument(
    "-pre_tgt",
    "--pretrained_tgt",
    help="path to src pretrained vectors",
    default='none'
)
parser.add_argument(
    "-depth",
    "--num_encoder_layers",
    help="Depth of the encoder",
    default='1'
)
parser.add_argument(
    "-load",
    "--load_model",
    help="Epoch number of the model to be loaded",
    default='none'
)
args = parser.parse_args()

data_path_train_src = args.train_src_sentences
data_path_train_tgt = args.train_tgt_sentences
data_path_dev_src = args.dev_src_sentences
data_path_dev_tgt = args.dev_tgt_sentences
batch_size = int(args.batch_size)
experiment_name = args.experiment_name

np.random.seed(seed=int(args.seed))  # set seed for an experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def get_vocab(lines):
    """Get vocabulary for a language."""
    vocab = set()
    for line in lines:
        for word in line:
            vocab.add(word)
    vocab.add('<s>')
    vocab.add('</s>')
    vocab.add('<unk>')
    word2ind = {}
    ind2word = {}

    for ind, word in enumerate(vocab):
        word2ind[word] = ind
        ind2word[ind] = word
    return vocab, word2ind, ind2word


def softmax_3d(x):
    """Compute softmax on 3D tensor by broadcasting first dim."""
    e = T.exp(x - T.max(x, axis=-1, keepdims=True))
    s = T.sum(e, axis=-1, keepdims=True)
    return e / s


def prepare_batch(src_sentences, tgt_sentences):
    """Prepare a mini-batch for training."""
    src_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in src_sentences]
    tgt_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in tgt_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    tgt_lens = [len(sent) for sent in tgt_sentences]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    src_sentences = [
        [src_word2ind[word] if word in src_word2ind else src_word2ind['<unk>'] for word in sent] +
        [src_word2ind['<unk>']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    tgt_sentences_inp = [
        [tgt_word2ind[word] if word in tgt_word2ind else tgt_word2ind['<unk>'] for word in sent[:-1]] +
        [tgt_word2ind['<unk>']] * (max_tgt_len - len(sent))
        for sent in tgt_sentences
    ]
    tgt_sentences_op = [
        [tgt_word2ind[word] if word in tgt_word2ind else tgt_word2ind['<unk>'] for word in sent[1:]] +
        [tgt_word2ind['<unk>']] * (max_tgt_len - len(sent))
        for sent in tgt_sentences
    ]
    tgt_mask = np.array(
        [
            ([1] * (l - 1)) + ([0] * (max_tgt_len - l))
            for l in tgt_lens
        ]
    ).astype(np.float32)
    src_sentences = np.array(src_sentences).astype(np.int32)
    tgt_sentences_inp = np.array(tgt_sentences_inp).astype(np.int32)
    tgt_sentences_op = np.array(tgt_sentences_op).astype(np.int32)
    src_lens = np.array(src_lens).astype(np.int32)
    return src_sentences, tgt_sentences_inp, tgt_sentences_op, \
        src_lens, tgt_mask


def prepare_evaluation_batch(src_sentences):
    """Prepare a mini-batch for evaluation."""
    src_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in src_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    max_src_len = max(src_lens)
    src_sentences = [
        [src_word2ind[word] if word in src_word2ind else src_word2ind['<unk>'] for word in sent] +
        [src_word2ind['<unk>']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    src_sentences = np.array(src_sentences).astype(np.int32)
    return src_sentences, src_lens


def decode_batch(src_sentences):
    """Decode one mini-batch for source sentences."""
    src_sentences, src_lens = prepare_evaluation_batch(src_sentences)
    decode_state = np.array(
        [[tgt_word2ind['<s>']] for _ in src_sentences]
    ).astype(np.int32)
    is_decoding = [True] * len(src_sentences)
    decode_length = 1
    while any(is_decoding) and decode_length < 25:
        next_words = f_eval(
            src_sentences,
            decode_state,
            src_lens
        )
        next_words = [x[-1] for x in np.argmax(next_words, axis=2)]
        is_finished = [word == tgt_word2ind['</s>'] for word in next_words]
        for ind, item in enumerate(is_finished):
            if item:
                is_decoding[ind] = False
        decode_state = np.c_[decode_state, next_words].astype(np.int32)
        decode_length += 1
    return decode_state


def decode_dev():
    """Decode the entire dev set by chunking into mini-batches."""
    decoded_sentences = []
    for i in range(0, len(dev_src), batch_size):
        logging.info('Decoding batch %d out of %d ' % (i, len(dev_src)))
        decoded_batch = decode_batch(dev_src[i:i + batch_size])
        decoded_batch = [
            [tgt_ind2word[x] for x in sent] for sent in decoded_batch
        ]
        trimmed_decoded_batch = []
        for sent in decoded_batch:
            stop_ind = sent.index('</s>') if '</s>' in sent else len(sent)
            trimmed_decoded_batch.append(sent[1:stop_ind])
        decoded_sentences += trimmed_decoded_batch
    return decoded_sentences


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x)/y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)


def get_validation_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def shuffle_data(src, tgt):
    """Shuffle source and target."""
    src, tgt = shuffle(src, tgt)
    return src, tgt


def save_model(epoch, minibatch, model_params):
    """Save the entire model."""
    if not os.path.exists('/data/lisatmp4/subramas/models/%s' % (
        experiment_name
    )):
        os.mkdir('/data/lisatmp4/subramas/models/%s' % (experiment_name))
    os.mkdir('/data/lisatmp4/subramas/models/%s/epoch_%d_minibatch_%d' % (
        experiment_name,
        epoch,
        minibatch
    ))
    model_path = '/data/lisatmp4/subramas/models/%s/epoch_%d_minibatch_%d' % (
        experiment_name,
        epoch,
        minibatch
    )
    logging.info('Saving word embeddings ...')
    for param in model_params:
        logging.info('Saving parameter %s ' % (param.name))
        np.save('%s/%s' % (model_path, param.name), param.get_value())
    pickle.dump(
        src_word2ind, open('%s/%s' % (model_path, 'src_word2ind'), 'wb')
    )
    pickle.dump(
        src_ind2word, open('%s/%s' % (model_path, 'src_ind2word'), 'wb')
    )
    pickle.dump(
        tgt_word2ind, open('%s/%s' % (model_path, 'tgt_word2ind'), 'wb')
    )
    pickle.dump(
        tgt_ind2word, open('%s/%s' % (model_path, 'tgt_ind2word'), 'wb')
    )


def load_model(epoch, minibatch, model_params):
    """Load a model's parameters."""
    model_path = '/data/lisatmp4/subramas/models/%s/epoch_%d_minibatch_%d' % (
        experiment_name,
        epoch,
        minibatch
    )
    assert len(model_params) == len(os.listdir(model_path))
    for param in model_params:
        assert os.path.exists('%s/%s.npy' % (model_path, param.name))
    for param in model_params:
        value = np.load('%s/%s.npy' % (model_path, param.name))
        param.set_value(value)

    src_word2ind = pickle.load(
        open('%s/%s' % (model_path, 'src_word2ind'), 'rb')
    )
    src_ind2word = pickle.load(
        open('%s/%s' % (model_path, 'src_ind2word'), 'rb')
    )
    tgt_word2ind = pickle.load(
        open('%s/%s' % (model_path, 'tgt_word2ind'), 'rb')
    )
    tgt_ind2word = pickle.load(
        open('%s/%s' % (model_path, 'tgt_ind2word'), 'rb')
    )

    return src_word2ind, src_ind2word, tgt_word2ind, tgt_ind2word


def str2float(line):
    """List of string numbers to float array."""
    return np.array(line).astype(np.float32)


def get_pretrained_embedding_layer(file_path, vocab, src_tgt):
    """Create a pretrained emebdding lookup table."""
    pretrained_lines = [line.strip().split() for line in codecs.open(
        file_path,
        'r',
        encoding='utf-8'
    )]
    word2ind = {}
    ind2word = {}
    ind = 0
    emebdding_dim = int(pretrained_lines[0][1])
    pretrained_lines = {
        line[0]: str2float(line[1:]) for line in pretrained_lines[1:]
        if len(line) == emebdding_dim + 1
    }
    pretrained_embedding = np.random.rand(
        len(vocab),
        emebdding_dim
    ).astype(np.float32)
    for ind, word in enumerate(vocab):
        word2ind[word] = ind
        ind2word[ind] = word
        if word in pretrained_lines:
            pretrained_embedding[ind] = pretrained_lines[word]
        elif word == '<unk>':
            pretrained_embedding[ind] = pretrained_lines['unk']
    embedding_layer = EmbeddingLayer(
        input_dim=pretrained_embedding.shape[0],
        output_dim=pretrained_embedding.shape[1],
        pretrained=pretrained_embedding,
        name='pretrained_embedding_%s' % (src_tgt)
    )
    return embedding_layer, word2ind, ind2word


def print_decoded_dev(dev_predictions, n=10):
    """Print the first n decoded sentences."""
    logging.info('Printing decoded dev sentences ...')
    for i in xrange(n):
        logging.info(' '.join(dev_predictions[i]))


def generate_samples(
    batch_src,
    batch_tgt_inp,
    batch_src_lens,
):
    """Generate random samples."""
    decoded_batch = f_eval(
        batch_src,
        batch_tgt_inp,
        batch_src_lens,
    )
    decoded_batch = np.argmax(decoded_batch, axis=2)
    for ind, sentence in enumerate(decoded_batch[:10]):
        logging.info('Src : %s ' % (' '.join([
            src_ind2word[x] for x in batch_src[ind]]
        )))
        logging.info('Tgt : %s ' % (' '.join([
            tgt_ind2word[x] for x in batch_tgt_inp[ind]]
        )))
        logging.info('Sample : %s ' % (' '.join([
            tgt_ind2word[x] for x in decoded_batch[ind]]
        )))

# Read training and dev data
train_src = [line.strip().split() for line in codecs.open(
    data_path_train_src,
    'r',
    encoding='utf-8'
)]
train_tgt = [line.strip().split() for line in codecs.open(
    data_path_train_tgt,
    'r',
    encoding='utf-8'
)]

dev_src = [line.strip().lower().split() for line in codecs.open(
    data_path_dev_src,
    'r',
    encoding='utf-8'
)]
dev_tgt = [line.strip().lower().split() for line in codecs.open(
    data_path_dev_tgt,
    'r',
    encoding='utf-8'
)]

# Get training and dev vocabularies
src_vocab, src_word2ind, src_ind2word = get_vocab(train_src)
tgt_vocab, tgt_word2ind, tgt_ind2word = get_vocab(train_tgt)

logging.info('Running experiment with seed %s ...' % (args.seed))

logging.info('Finished reading data ...')
logging.info('Number of training sentence-pairs : %d ' % (len(train_src)))
logging.info('Number of validation sentence-pairs : %d ' % (len(dev_src)))

# Create symbolic variables
src_inp = T.imatrix()
tgt_inp = T.imatrix()
tgt_op = T.imatrix()
src_lens = T.ivector()
tgt_mask = T.fmatrix()
index = T.scalar()

# Create synthetic data to test computation graph
src_inp_t = np.random.randint(low=1, high=100, size=(5, 10)).astype(np.int32)
tgt_inp_t = np.random.randint(low=1, high=100, size=(5, 15)).astype(np.int32)
tgt_op_t = np.random.randint(low=1, high=200, size=(5, 15)).astype(np.int32)
src_lens_t = np.random.randint(low=1, high=9, size=(5,)).astype(np.int32)
tgt_mask_t = np.float32(np.random.rand(5, 15).astype(np.float32) > 0.5)

# Embedding Lookup Tables
if args.pretrained_src != 'none':
    logging.info('Reading src pretrained embeddings ...')
    src_embedding_layer, src_word2ind, src_ind2word \
        = get_pretrained_embedding_layer(args.pretrained_src, src_vocab, 'src')
    n_src = len(src_word2ind)  # number of words in the source language

else:
    n_src = len(src_word2ind)  # number of words in the source language
    src_embedding_layer = EmbeddingLayer(
        input_dim=n_src,
        output_dim=src_emb_dim,
        name='src_embedding'
    )

if args.pretrained_tgt != 'none':
    logging.info('Reading tgt pretrained embeddings ...')
    tgt_embedding_layer, tgt_word2ind, tgt_ind2word \
        = get_pretrained_embedding_layer(args.pretrained_tgt, tgt_vocab, 'tgt')
    n_tgt = len(tgt_word2ind)
else:
    n_tgt = len(tgt_word2ind)  # number of words in the source language
    tgt_embedding_layer = EmbeddingLayer(
        input_dim=n_tgt,
        output_dim=tgt_emb_dim,
        name='tgt_embedding'
    )

logging.info('Source vocabulary size : %d ' % (
    src_embedding_layer.input_dim
))
logging.info('Target vocabulary size : %d ' % (
    tgt_embedding_layer.input_dim
))

if args.pretrained_src != 'none':
    src_emb_dim = src_embedding_layer.output_dim
if args.pretrained_tgt != 'none':
    tgt_emb_dim = tgt_embedding_layer.output_dim

# Encoder BiLSTM and Decoder LSTM
encoder_forward = [
    GRU(
        input_dim=src_emb_dim,
        output_dim=src_lstm_op_dim,
        batch_input=True,
        name='src_lstm_forward_0'
    )
]
encoder_backward = [
    GRU(
        input_dim=src_emb_dim,
        output_dim=src_lstm_op_dim,
        batch_input=True,
        name='src_lstm_backward_0'
    )
]
# Make the LSTM deep
for i in range(int(args.num_encoder_layers) - 1):
    encoder_forward.append(
        GRU(
            input_dim=2 * src_lstm_op_dim,
            output_dim=src_lstm_op_dim,
            batch_input=True,
            name='src_lstm_forward_%d' % (i)
        )
    )
    encoder_backward.append(
        GRU(
            input_dim=2 * src_lstm_op_dim,
            output_dim=src_lstm_op_dim,
            batch_input=True,
            name='src_lstm_backward_%d' % (i)
        )
    )
tgt_lstm = GRU(
    input_dim=tgt_emb_dim,
    output_dim=tgt_lstm_op_dim,
    batch_input=True,
    name='tgt_lstm'
)

# Projection layers
tgt_lstm_h_to_vocab = FullyConnectedLayer(
    input_dim=tgt_lstm_op_dim + 2 * src_lstm_op_dim,
    output_dim=n_tgt,
    batch_normalization=False,
    activation='softmax',
    name='tgt_lstm_h_to_vocab'
)

encoder_decoder_projection = FullyConnectedLayer(
    input_dim=2 * src_lstm_op_dim,
    output_dim=tgt_lstm_op_dim,
    batch_normalization=False,
    activation='tanh',
    name='encoder_decoder_connection'
)

if args.attention == 'mlp':
    attention_layer_1 = FullyConnectedLayer(
        input_dim=tgt_lstm_op_dim,
        output_dim=tgt_lstm_op_dim,
        batch_normalization=False,
        activation='relu',
        name='attention_layer_1'
    )

    attention_layer_2 = FullyConnectedLayer(
        input_dim=tgt_lstm_op_dim,
        output_dim=tgt_lstm_op_dim,
        batch_normalization=False,
        activation='relu',
        name='attention_layer_2'
    )

# Set model parameters
params = src_embedding_layer.params + tgt_embedding_layer.params + \
    encoder_decoder_projection.params
for rnn in encoder_forward + encoder_backward:
    params += rnn.params
params += tgt_lstm.params[:-1] + tgt_lstm_h_to_vocab.params

if args.attention == 'mlp':
    params += attention_layer_1.params + attention_layer_2.params

logging.info('Model parameters ...')
logging.info('Src Embedding dim : %d ' % (src_emb_dim))
logging.info('Tgt Embedding dim : %d ' % (tgt_emb_dim))
logging.info('Encoder BiLSTM dim : %d ' % (2 * encoder_forward[-1].output_dim))
logging.info('Batch size : %s ' % (batch_size))
logging.info('Decoder LSTM dim : %d ' % (tgt_lstm.output_dim))
logging.info('Attention mechanism : %s ' % (args.attention))
logging.info('Depth : %s ' % (args.num_encoder_layers))

# Get embedding matrices
src_emb_inp = src_embedding_layer.fprop(src_inp)
tgt_emb_inp = tgt_embedding_layer.fprop(tgt_inp)

encoder_representation = None
# Get BiLSTM representations
for ind, (forward_rnn, backward_rnn) in enumerate(
    zip(encoder_forward, encoder_backward)
):
    if ind == 0:
        forward_rnn.fprop(src_emb_inp[::-1, :])
        backward_rnn.fprop(src_emb_inp)
    else:
        forward_rnn.fprop(encoder_representation[::-1, :])
        backward_rnn.fprop(encoder_representation)
    encoder_representation = T.concatenate(
        (forward_rnn.h, backward_rnn.h[::-1, :]),
        axis=2
    )

encoder_final_state = encoder_representation.dimshuffle(1, 0, 2)[
    T.arange(src_inp.shape[0]), src_lens - 1, :
]

# Get Target LSTM representation & Attention Vectors
tgt_lstm.h_0 = encoder_decoder_projection.fprop(encoder_final_state)
tgt_lstm.fprop(tgt_emb_inp)

# Attention
if args.attention == 'mlp':
    attention = attention_layer_1.fprop(tgt_lstm.h.dimshuffle(1, 0, 2))
    attention = attention_layer_2.fprop(attention)
    attention = T.batched_dot(
        attention,
        encoder_representation.dimshuffle(1, 2, 0)
    )
    attention = T.batched_dot(
        attention,
        encoder_representation.dimshuffle(1, 0, 2)
    )
elif args.attention == 'dot':
    attention = T.batched_dot(
        tgt_lstm.h.dimshuffle(1, 0, 2),
        encoder_representation.dimshuffle(1, 2, 0)
    )
    attention = T.batched_dot(
        attention,
        encoder_representation.dimshuffle(1, 0, 2)
    )

# Concatenate the attention vectors to the Target LSTM output
# before predicting the next word
target_representation = T.concatenate(
    [attention, tgt_lstm.h.dimshuffle(1, 0, 2)],
    axis=2
)

# Project to target vocabulary and softmax
proj_layer_input = target_representation.reshape(
    (tgt_inp.shape[0] * tgt_inp.shape[1], target_representation.shape[2])
)
proj_output_rep = tgt_lstm_h_to_vocab.fprop(proj_layer_input)
final_output = proj_output_rep.reshape(
    (tgt_inp.shape[0], tgt_inp.shape[1], n_tgt)
)
# Clip final out to avoid log problem in cross-entropy
final_output = T.clip(final_output, 1e-5, 1-1e-5)

# Compute cost
cost = - (T.log(final_output[
    T.arange(
        tgt_emb_inp.shape[0]).dimshuffle(0, 'x').repeat(
            tgt_emb_inp.shape[1],
            axis=1
        ).flatten(),
    T.arange(tgt_emb_inp.shape[1]).dimshuffle('x', 0).repeat(
        tgt_emb_inp.shape[0],
        axis=0
    ).flatten(),
    tgt_op.flatten()
]) * tgt_mask.flatten()).sum() / T.neq(tgt_mask, 0).sum()

logging.info('Computation Graph Node Shapes ...')
logging.info('src embedding dim : %s ' % (
    src_emb_inp.eval({src_inp: src_inp_t}).shape,)
)
logging.info('tgt embedding dim : %s' % (
    tgt_emb_inp.eval({tgt_inp: tgt_inp_t}).shape,)
)
logging.info('encoder forward dim : %s' % (
    encoder_forward[0].h.dimshuffle(1, 0, 2).eval({src_inp: src_inp_t}).shape,)
)
logging.info('encoder backward dim : %s' % (
    encoder_backward[0].h.dimshuffle(1, 0, 2).eval(
        {src_inp: src_inp_t}).shape,)
)
logging.info('encoder hidden state shape : %s' % (
    encoder_representation.eval({src_inp: src_inp_t}).shape,)
)
logging.info('encoder final state shape : %s' % (encoder_final_state.eval(
    {src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('decoder hidden state shape : %s' % (
    tgt_lstm.h.dimshuffle(1, 0, 2).eval(
        {src_inp: src_inp_t, tgt_inp: tgt_inp_t, src_lens: src_lens_t}
    ).shape,)
)
logging.info('attention vectors shape : %s' % (attention.eval(
    {tgt_inp: tgt_inp_t, src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('decoder + attention shape : %s' % (target_representation.eval(
    {tgt_inp: tgt_inp_t, src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('tgt vocab projection shape : %s' % (proj_output_rep.eval(
    {tgt_inp: tgt_inp_t, src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('tgt vocab softmax shape : %s' % (final_output.eval(
    {tgt_inp: tgt_inp_t, src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('cost : %.3f' % (cost.eval(
    {
        tgt_inp: tgt_inp_t,
        src_inp: src_inp_t,
        tgt_op: tgt_op_t,
        src_lens: src_lens_t,
        tgt_mask: tgt_mask_t
    }
)))

logging.info('Compiling theano functions ...')

updates = Optimizer(clip=5.0).adam(
    cost=cost,
    params=params,
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


if args.load_model != 'none':
    src_word2ind, src_ind2word, tgt_word2ind, tgt_ind2word \
        = load_model(int(args.load_model), params)

dev_tgt = [
    [word if word in tgt_word2ind else '<unk>' for word in sent]
    for sent in dev_tgt
]

epoch_offset = 0 if args.load_model == 'none' else int(args.load_model)
num_epochs = 100
logging.info('Training network ...')
BEST_BLEU = 1.0
costs = []
for i in range(num_epochs):
    logging.info('Shuffling data ...')
    train_src, train_tgt = shuffle_data(train_src, train_tgt)
    for j in xrange(0, len(train_src), batch_size):
        batch_src, batch_tgt_inp, batch_tgt_op, batch_src_lens, batch_tgt_mask \
            = prepare_batch(
                train_src[j: j + batch_size], train_tgt[j: j + batch_size]
            )
        entropy = f_train(
            batch_src,
            batch_tgt_inp,
            batch_tgt_op,
            batch_src_lens,
            batch_tgt_mask
        )
        costs.append(entropy)
        logging.info('Epoch : %d Minibatch : %d Loss : %.3f' % (
            i + epoch_offset,
            j,
            entropy
        ))
        if j % 64000 == 0 and j != 0:
            dev_predictions = decode_dev()
            dev_bleu = get_validation_bleu(dev_predictions, dev_tgt)
            if dev_bleu > BEST_BLEU:
                BEST_BLEU = dev_bleu
                print_decoded_dev(dev_predictions)
                save_model(i + epoch_offset, j, params)
            logging.info('Epoch : %d Minibatch :%d dev BLEU : %.3f' % (
                i + epoch_offset,
                j,
                dev_bleu)
            )
            logging.info('Mean Cost : %.3f' % (np.mean(costs)))
            costs = []
        if j % 6400 == 0:
            generate_samples(batch_src, batch_tgt_inp, batch_src_lens)
    dev_predictions = decode_dev()
    dev_bleu = get_validation_bleu(dev_predictions, dev_tgt)
    if dev_bleu > BEST_BLEU:
        BEST_BLEU = dev_bleu
        print_decoded_dev(dev_predictions)
        save_model(i + epoch_offset, j, params)
    logging.info('Epoch : %d dev BLEU : %.3f' % (
        i + epoch_offset,
        dev_bleu)
    )
    logging.info('Mean Cost : %.3f' % (np.mean(costs)))
    costs = []
