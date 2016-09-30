"""Functions to save and load models."""
import os
import pickle
import numpy as np
import codecs
import sys

sys.path.append('/home/subras/Research/SanDeepLearn/')

from recurrent import FastLSTM


def str2float(line):
    """List of string numbers to float array."""
    return np.array(line).astype(np.float32)


def save_model(
    epoch,
    minibatch,
    model_params,
    experiment_name,
    src_word2ind,
    tgt_word2ind,
    src_ind2word,
    tgt_ind2word
):
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
    for param in model_params:
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


def load_model(name, model_params):
    """Load a model's parameters."""
    parent_folder = name.split('/')[0]
    child_folder = name.split('/')[1]
    model_path = '/data/lisatmp4/subramas/models/%s/%s' % (
        parent_folder,
        child_folder
    )
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


def load_lstm(folder_path, layer_num):
    """Load LSTM weights and bias."""
    lstm_W = np.load(open(
        '%s/%s' % (folder_path, 'lstm_W_%d.npy' % (layer_num)),
        'rb'
    ))
    lstm_U = np.load(open(
        '%s/%s' % (folder_path, 'lstm_U_%d.npy' % (layer_num)),
        'rb'
    ))
    lstm_b = np.load(open(
        '%s/%s' % (folder_path, 'lstm_b_%d.npy' % (layer_num)),
        'rb'
    ))
    return lstm_W, lstm_U, lstm_b


def set_lstm_params(lstm, lstm_W, lstm_U, lstm_b):
    """Set parameters of an LSTM."""
    lstm.W.set_value(lstm_W)
    lstm.U.set_value(lstm_U)
    lstm.b.set_value(lstm_b)


def load_tf_model(folder_path):
    """Load tensorflow model."""
    assert os.path.exists(folder_path)

    lstm_0 = FastLSTM(
        input_dim=1024,
        output_dim=1024,
        name='lstm_0'
    )
    lstm_1 = FastLSTM(
        input_dim=1024,
        output_dim=1024,
        name='lstm_1'
    )
    lstm_2 = FastLSTM(
        input_dim=1024,
        output_dim=1024,
        name='lstm_2'
    )

    lstm_0_W, lstm_0_U, lstm_0_b = load_lstm(folder_path, 0)
    lstm_1_W, lstm_1_U, lstm_1_b = load_lstm(folder_path, 1)
    lstm_2_W, lstm_2_U, lstm_2_b = load_lstm(folder_path, 2)

    set_lstm_params(lstm_0, lstm_0_W, lstm_0_U, lstm_0_b)
    set_lstm_params(lstm_1, lstm_1_W, lstm_1_U, lstm_1_b)
    set_lstm_params(lstm_2, lstm_2_W, lstm_2_U, lstm_2_b)

    src_embedding = np.load(open(
        '%s/%s' % (folder_path, 'src_embedding.npy'),
        'rb'
    ))

    return src_embedding, lstm_0, lstm_1, lstm_2


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
