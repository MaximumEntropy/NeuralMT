"""Set model parameters to be used for NMT."""
src_emb_dim = 300  # source word embedding dimension
tgt_emb_dim = 300  # target word embedding dimension
src_rnn_op_dim = 1024  # source LSTMs hidden dimension
tgt_rnn_op_dim = 1024  # target LSTM hidden dimension
ctx_dim = src_rnn_op_dim  # Size of the context
beta = 500  # Regularization coefficient
