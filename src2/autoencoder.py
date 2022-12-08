import numpy as np
import torch
from sequitur.models import LSTM_AE, CONV_LSTM_AE
from sequitur import quick_train
from torch import FloatTensor
from torch.nn import MSELoss
from sklearn.preprocessing import MinMaxScaler
from src import utils

org_train, org_test = utils.load_data("../processed_data/student_trajectories.pkl", test_size=.2)

scaler = MinMaxScaler()
scaler = scaler.fit(np.stack(org_train['state']))
states = FloatTensor(utils.pad_states(org_train, scaler=scaler))
test_states = FloatTensor(utils.pad_states(org_test, scaler=scaler))


model = LSTM_AE(
  input_dim=states.shape[-1],
  encoding_dim=256,
  h_dims=[32],
  h_activ=None,
  out_activ=None
)

def train_autoencoder(model, steps, dryrun):
    if force_train is False:
        is_loaded = self.load()
        if is_loaded:
            return

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = MSELoss(size_average=False)
    mean_losses = []

    for epoch in range(100):
        model.train()

        losses = []
        for x in states:
            optimizer.zero_grad()

            # Forward pass
            x_prime = model(x)

            loss = criterion(x_prime, x)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        mean_loss = np.mean(losses)
        mean_losses.append(mean_loss)

        print(f"Epoch: {epoch}, Loss: {mean_loss}")

    torch.save(model.state_dict(), "../checkpoint/autoencoder_test.ckpt")
