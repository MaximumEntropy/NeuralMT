"""Set model parameters to be used for NMT."""
src_emb_dim = 256  # source word embedding dimension
tgt_emb_dim = 256  # target word embedding dimension
src_lstm_op_dim = 512  # source LSTMs hidden dimension
tgt_lstm_op_dim = 2 * src_lstm_op_dim  # target LSTM hidden dimension
beta = 500  # Regularization coefficient
