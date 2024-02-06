class BLSTM_Decoder(nn.Module):

  def __init__(self, seq_len, n_features=1, input_dim=64,):
    super(BLSTM_Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 4 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=self.input_dim * 4,  #
      hidden_size=self.input_dim * 2,
      num_layers=1,
      batch_first=True)

    self.rnn2 = nn.LSTM(
      input_size=self.input_dim * 2,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True)

    self.output_layer = nn.Linear(self.hidden_dim,self.n_features) #

  def forward(self, x):
      x = x.reshape((-1,1,self.input_dim*4)) #
      x = x.repeat(1,self.seq_len,1)
      x, _ = self.rnn1(x)
      x, _ = self.rnn2(x)
      return self.output_layer(x)