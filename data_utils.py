"""Data utilities."""
import numpy as np
from sklearn.utils import shuffle


def get_src_mask(src_lens, max_src_len):
    """Get a mask for the src minibatch."""
    return np.array(
        [
            ([1] * (l - 1)) + ([0] * (max_src_len - l))
            for l in src_lens
        ]
    ).transpose().astype(np.float32)


def prepare_batch(src_sentences, tgt_sentences, src_word2ind, tgt_word2ind):
    """Prepare a mini-batch for training."""
    src_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in src_sentences]
    tgt_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in tgt_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    tgt_lens = [len(sent) for sent in tgt_sentences]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    src_sentences = [
        [src_word2ind[w] for w in sent] +
        [src_word2ind['<pad>']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    tgt_sentences_inp = [
        [tgt_word2ind[w] for w in sent[:-1]] +
        [tgt_word2ind['<pad>']] * (max_tgt_len - len(sent))
        for sent in tgt_sentences
    ]
    tgt_sentences_op = [
        [tgt_word2ind[w] for w in sent[1:]] +
        [tgt_word2ind['<unk>']] * (max_tgt_len - len(sent))
        for sent in tgt_sentences
    ]
    tgt_mask = np.array(
        [
            ([1] * (l - 1)) + ([0] * (max_tgt_len - l))
            for l in tgt_lens
        ]
    ).astype(np.float32)
    src_mask = get_src_mask(src_lens, max_src_len)
    src_sentences = np.array(src_sentences).astype(np.int32)
    tgt_sentences_inp = np.array(tgt_sentences_inp).astype(np.int32)
    tgt_sentences_op = np.array(tgt_sentences_op).astype(np.int32)
    src_lens = np.array(src_lens).astype(np.int32)
    return src_sentences, tgt_sentences_inp, tgt_sentences_op, \
        src_lens, src_mask, tgt_mask


def prepare_autoencode_batch(
    src_sentences,
    src_word2ind
):
    """Prepare a mini-batch for training."""
    src_sentences = [['_GO'] + sent[:40] + ['_EOS'] for sent in src_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    max_src_len = max(src_lens)
    src_sentences = [
        [
            src_word2ind[w] if w in src_word2ind else src_word2ind['_UNK']
            for w in sent
        ] +
        [src_word2ind['_PAD']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    tgt_mask = np.array(
        [
            ([1] * (l - 1)) + ([0] * (max_src_len - l))
            for l in src_lens
        ]
    ).astype(np.float32)
    src_sentences = np.array(src_sentences).astype(np.int32)
    src_lens = np.array(src_lens).astype(np.int32)
    return src_sentences, src_lens, tgt_mask


def prepare_bucketed_batch(minibatch, src_word2ind, tgt_word2ind):
    """Prepare a mini-batch for training."""
    src_sentences = [x[0] for x in minibatch]
    tgt_sentences = [x[1] for x in minibatch]
    src_sentences = [['<s>'] + sent + ['</s>'] for sent in src_sentences]
    tgt_sentences = [['<s>'] + sent + ['</s>'] for sent in tgt_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    max_src_len = max(src_lens)
    src_sentences = [
        [
            src_word2ind[w] if w in src_word2ind else src_word2ind['<unk>']
            for w in sent
        ] +
        [src_word2ind['<unk>']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    tgt_sentences_inp = [
        [
            tgt_word2ind[w] if w in tgt_word2ind else tgt_word2ind['<unk>']
            for w in sent[:-1]
        ]
        for sent in tgt_sentences
    ]
    tgt_sentences_op = [
        [
            tgt_word2ind[w] if w in tgt_word2ind else tgt_word2ind['<unk>']
            for w in sent[1:]
        ]
        for sent in tgt_sentences
    ]
    src_sentences = np.array(src_sentences).astype(np.int32)
    tgt_sentences_inp = np.array(tgt_sentences_inp).astype(np.int32)
    tgt_sentences_op = np.array(tgt_sentences_op).astype(np.int32)
    src_lens = np.array(src_lens).astype(np.int32)
    return src_sentences, tgt_sentences_inp, tgt_sentences_op, \
        src_lens


def prepare_evaluation_batch(src_sentences, src_word2ind):
    """Prepare a mini-batch for evaluation."""
    src_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in src_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    max_src_len = max(src_lens)
    src_sentences = [
        [
            src_word2ind[w] if w in src_word2ind else src_word2ind['<unk>']
            for w in sent
        ] +
        [src_word2ind['<pad>']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    src_mask = get_src_mask(src_lens, max_src_len)
    src_sentences = np.array(src_sentences).astype(np.int32)
    return src_sentences, src_lens, src_mask


def get_vocab(lines):
    """Get vocabulary for a language."""
    vocab = set()
    for line in lines:
        for word in line:
            vocab.add(word)
    vocab.add('<s>')
    vocab.add('</s>')
    vocab.add('<unk>')
    vocab.add('<pad>')
    word2ind = {}
    ind2word = {}

    for ind, word in enumerate(vocab):
        word2ind[word] = ind
        ind2word[ind] = word
    return vocab, word2ind, ind2word


def append_to_vocab(
    lines,
    word2ind,
    ind2word,
    embedding
):
    """Append new vocabulary to pretrained vocab."""
    max_ind = max(ind2word.keys()) + 1
    ind = max_ind
    for line in lines:
        for word in line:
            if word not in word2ind:
                word2ind[word] = ind
                ind2word[ind] = word
                ind += 1
    diff = ind - max_ind
    additional_embeddings = np.random.rand(diff, embedding.shape[1])
    embedding = np.vstack((embedding, additional_embeddings)).astype(np.float32)
    assert embedding.shape[0] == len(word2ind) == len(ind2word)
    return word2ind, ind2word, embedding


def shuffle_data(src, tgt):
    """Shuffle source and target."""
    src, tgt = shuffle(src, tgt)
    return src, tgt
