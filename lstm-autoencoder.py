"""LSTM sequence-sequence autoencoder."""
import theano
import theano.tensor as T
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import codecs
import argparse
import logging

from model import load_tf_model
from bleu import get_bleu
from data_utils import prepare_autoencode_batch, prepare_evaluation_batch, \
    shuffle_data, append_to_vocab

sys.path.append('/u/subramas/Research/SanDeepLearn/')

from recurrent import FastLSTM
from layer import FullyConnectedLayer, EmbeddingLayer
from optimizers import Optimizer

theano.config.floatX = 'float32'

parser = argparse.ArgumentParser()
parser.add_argument(
    "-train_src",
    "--train_src_sentences",
    help="path to train src sentences",
    required=True
)
parser.add_argument(
    "-dev_src",
    "--dev_src_sentences",
    help="path to dev src sentences",
    required=True
)
parser.add_argument(
    "-src_vocab",
    "--src_vocab_path",
    help="path to src vocabulary",
    required=True
)
parser.add_argument(
    "-batch_size",
    "--batch_size",
    help="batch size",
    required=True
)
parser.add_argument(
    "-epath",
    "--encoder_path",
    help="path to encoder",
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
args = parser.parse_args()
data_path_train_src = args.train_src_sentences
data_path_dev_src = args.dev_src_sentences
np.random.seed(seed=int(args.seed))  # set seed for an experiment
experiment_name = args.experiment_name
batch_size = args.batch_size

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


def decode_batch(src_sentences):
    """Decode one mini-batch for source sentences."""
    src_sentences, src_lens = prepare_evaluation_batch(
        src_sentences,
        src_word2ind
    )
    decode_state = np.array(
        [[src_word2ind['<s>']] for _ in src_sentences]
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
        is_finished = [word == src_word2ind['</s>'] for word in next_words]
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
            [src_word2ind[x] for x in sent] for sent in decoded_batch
        ]
        trimmed_decoded_batch = []
        for sent in decoded_batch:
            stop_ind = sent.index('</s>') if '</s>' in sent else len(sent)
            trimmed_decoded_batch.append(sent[1:stop_ind])
        decoded_sentences += trimmed_decoded_batch
    return decoded_sentences


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
            src_word2ind[x] for x in batch_tgt_inp[ind]]
        )))
        logging.info('Sample : %s ' % (' '.join([
            src_word2ind[x] for x in decoded_batch[ind]]
        )))
        logging.info('=======================================================')

# Read training and dev data
train_src = [line.strip().split() for line in codecs.open(
    data_path_train_src,
    'r',
    encoding='utf-8'
)]

dev_src = [line.strip().lower().split() for line in codecs.open(
    data_path_dev_src,
    'r',
    encoding='utf-8'
)]

# Load tf model
logging.info('Loading Tensorflow model ...')
src_embedding, src_lstm_0, src_lstm_1, src_lstm_2 = load_tf_model(
    args.encoder_path
)

words = [line.strip() for line in codecs.open(
    args.src_vocab_path,
    'rb',
    encoding='utf-8'
    )
]

src_word2ind = {word: ind for ind, word in enumerate(words)}
src_ind2word = {ind: word for ind, word in enumerate(words)}

# Get training and dev vocabularies
logging.info('Getting vocabularies ...')
src_word2ind, src_ind2word, src_embedding = append_to_vocab(
    train_src,
    src_word2ind,
    src_ind2word,
    src_embedding
)

logging.info('Running experiment with seed %s ...' % (args.seed))

logging.info('Finished reading data ...')
logging.info('Number of training sentence-pairs : %d ' % (len(train_src)))
logging.info('Number of validation sentence-pairs : %d ' % (len(dev_src)))

# Create symbolic variables
src_inp = T.imatrix()
src_lens = T.ivector()
tgt_mask = T.fmatrix()
index = T.scalar()

# Create synthetic data to test computation graph
src_inp_t = np.random.randint(low=1, high=100, size=(5, 10)).astype(np.int32)
src_lens_t = np.random.randint(low=1, high=9, size=(5,)).astype(np.int32)
tgt_mask_t = np.float32(np.random.rand(5, 15).astype(np.float32) > 0.5)

src_embedding_layer = EmbeddingLayer(
    input_dim=src_embedding.shape[0],
    output_dim=src_embedding.shape[1],
    pretrained=src_embedding,
    name='src_embedding'
)

tgt_embedding_layer = EmbeddingLayer(
    input_dim=src_embedding.shape[0],
    output_dim=src_embedding.shape[1],
    name='tgt_embedding'
)

tgt_lstm_0 = FastLSTM(
    input_dim=1024,
    output_dim=1024,
    name='tgt_lstm_0'
)

tgt_lstm_1 = FastLSTM(
    input_dim=1024,
    output_dim=1024,
    name='tgt_lstm_1'
)

tgt_lstm_2 = FastLSTM(
    input_dim=1024,
    output_dim=1024,
    name='tgt_lstm_2'
)

tgt_lstm_h_to_vocab = FullyConnectedLayer(
    input_dim=1024,
    output_dim=tgt_embedding_layer.input_dim,
    batch_normalization=False,
    activation='softmax',
    name='tgt_lstm_h_to_vocab'
)

# Set model parameters
params = tgt_embedding_layer.params
params += [
    src_lstm_0.h_0, src_lstm_0.c_0, src_lstm_1.h_0, src_lstm_1.c_0,
    src_lstm_2.h_0, src_lstm_2.c_0
]

for rnn in [tgt_lstm_0, tgt_lstm_1, tgt_lstm_2]:
    params += rnn.params

params += tgt_lstm_h_to_vocab.params

logging.info('Model parameters ...')
logging.info('Src Embedding dim : %d ' % (src_embedding_layer.output_dim))
logging.info('Tgt Embedding dim : %d ' % (tgt_embedding_layer.output_dim))
logging.info('Encoder dim : %d ' % (src_lstm_2.output_dim))
logging.info('Batch size : %s ' % (batch_size))
logging.info('Decoder LSTM dim : %d ' % (tgt_lstm_2.output_dim))
logging.info('Depth : %s ' % ('3'))

# Get embedding matrices
src_emb_inp = src_embedding_layer.fprop(src_inp)
tgt_emb_inp = tgt_embedding_layer.fprop(src_inp[:, :-1])

# Get encoder representation
src_lstm_0.fprop(src_emb_inp)
src_lstm_1.fprop(src_lstm_0.h)
src_lstm_2.fprop(src_lstm_1.h)

encoder_final_state = src_lstm_2.h.dimshuffle(1, 0, 2)[
    T.arange(src_inp.shape[0]), src_lens - 1, :
]

# Connect encoder and decoder
tgt_lstm_0.h_0 = encoder_final_state

# Decode sentence from input
tgt_lstm_0.fprop(tgt_emb_inp)
tgt_lstm_1.fprop(tgt_lstm_0.h)
tgt_lstm_2.fprop(tgt_lstm_1.h)

# Project to target vocabulary and softmax
proj_layer_input = tgt_lstm_2.h.reshape(
    (src_inp.shape[0] * (src_inp.shape[1] - 1), tgt_lstm_2.h.shape[2])
)
proj_output_rep = tgt_lstm_h_to_vocab.fprop(proj_layer_input)
final_output = proj_output_rep.reshape(
    (src_inp.shape[0], (src_inp.shape[1] - 1), tgt_embedding_layer.input_dim)
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
    src_inp[:, 1:].flatten()
]) * tgt_mask.flatten()).sum() / T.neq(tgt_mask, 0).sum()

logging.info('Computation Graph Node Shapes ...')
logging.info('src embedding dim : %s ' % (
    src_emb_inp.eval({src_inp: src_inp_t}).shape,)
)
logging.info('tgt embedding dim : %s' % (
    tgt_emb_inp.eval({src_inp: src_inp_t}).shape,)
)
logging.info('encoder final state shape : %s' % (encoder_final_state.eval(
    {src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('decoder hidden state shape : %s' % (
    tgt_lstm_2.h.dimshuffle(1, 0, 2).eval(
        {src_inp: src_inp_t, src_lens: src_lens_t}
    ).shape,)
)

logging.info('tgt vocab projection shape : %s' % (proj_output_rep.eval(
    {src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('tgt vocab softmax shape : %s' % (final_output.eval(
    {src_inp: src_inp_t, src_lens: src_lens_t}).shape,)
)
logging.info('cost : %.3f' % (cost.eval(
    {
        src_inp: src_inp_t,
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
    inputs=[src_inp, src_lens, tgt_mask],
    outputs=cost,
    updates=updates
)

f_eval = theano.function(
    inputs=[src_inp, src_lens],
    outputs=final_output,
)
    