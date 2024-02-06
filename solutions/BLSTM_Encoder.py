class BLSTM_Encoder(nn.Module):

#Define a subclass of encoder that inherits from the parent class nn.Modul

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(BLSTM_Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

  # use to layers of LSTMs
    self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        bidirectional=True, # here to open bidirectional LSTM
        batch_first=True)

    self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim * 2, # the output of last layer doubled
        hidden_size=self.hidden_dim,
        num_layers=1,
        bidirectional=True,  # here to open bidirectional LSTM
        batch_first=True)


  def forward(self, x):
    x, _ = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    hidden_n = hidden_n.permute(1, 0, 2)
    # shape of hidden_n：(num_layers*num_directions, batch_size, embedding_dim)
    return hidden_n.reshape((-1,self.hidden_dim*2))