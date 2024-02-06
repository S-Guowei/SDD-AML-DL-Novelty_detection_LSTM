def train_model(model, train_loader, val_dataset, n_epochs, sigma=0):

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.L1Loss(reduction='sum').to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0

  for epoch in tqdm(range(1, n_epochs + 1),desc='Training'):
    model = model.train()
    train_losses = []
    for seq in train_loader:

      seq_true = seq[0]
      seq_true = seq_true.to(device)
      # add noise
      inputs = torch.randn_like(seq_true) * sigma * sigma + seq_true
      optimizer.zero_grad()
      inputs = inputs.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, seq_true)
      loss.backward()
      optimizer.step()

      train_losses.append(loss.item()/inputs.shape[0])

    val_losses = []
    model = model.eval()

    with torch.no_grad():
      for seq in val_dataset:

        seq_true = seq[0].unsqueeze(0)
        seq_true = seq_true.to(device)

        # add noise
        inputs = torch.randn_like(seq_true) * sigma * sigma + seq_true
        inputs = inputs.to(device)
        seq_pred = model(inputs)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())

    print(f' Train loss {train_loss} Validation loss {val_loss}')

  model.load_state_dict(best_model_wts)
  return model.eval(), history