import numpy as np
import torch
from sequitur.models import LSTM_AE, CONV_LSTM_AE
from sequitur import quick_train
from torch import FloatTensor
from torch.nn import MSELoss

from src import utils


def create_states(df, max_len=200):
    all_states = []
    for student, d in df.groupby('episode'):
        states = d['state']
        states = np.stack(states)
        curr_len = len(states)
        if curr_len <= max_len:
            pad_len = max_len - curr_len
            states = np.pad(states, pad_width=[(0, pad_len), (0, 0)], mode='constant', constant_values=0.)
        else:
            states = states[-max_len:, :]

        all_states.append(states)

    states = np.stack(all_states)
    return states


org_train, org_test = utils.load_data("../processed_data/student_trajectories.pkl", test_size=.2)

states = FloatTensor(create_states(org_train))

model = LSTM_AE(
  input_dim=108,
  encoding_dim=128,
  h_dims=[64],
  h_activ=None,
  out_activ=None
)

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

print(z)