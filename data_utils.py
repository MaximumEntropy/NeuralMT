"""Data utilities."""
import numpy as np
from sklearn.utils import shuffle


def prepare_batch(src_sentences, tgt_sentences, src_word2ind, tgt_word2ind):
    """Prepare a mini-batch for training."""
    src_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in src_sentences]
    tgt_sentences = [['<s>'] + sent[:40] + ['</s>'] for sent in tgt_sentences]
    src_lens = [len(sent) for sent in src_sentences]
    tgt_lens = [len(sent) for sent in tgt_sentences]
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
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
        ] +
        [tgt_word2ind['<unk>']] * (max_tgt_len - len(sent))
        for sent in tgt_sentences
    ]
    tgt_sentences_op = [
        [
            tgt_word2ind[w] if w in tgt_word2ind else tgt_word2ind['<unk>']
            for w in sent[1:]
        ] +
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
        [src_word2ind['<unk>']] * (max_src_len - len(sent))
        for sent in src_sentences
    ]
    src_sentences = np.array(src_sentences).astype(np.int32)
    return src_sentences, src_lens


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


def shuffle_data(src, tgt):
    """Shuffle source and target."""
    src, tgt = shuffle(src, tgt)
    return src, tgt