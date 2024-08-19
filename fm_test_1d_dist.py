from fm_model_simple import *
from torch.utils.data import TensorDataset, DataLoader

# Other tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from numpy.typing import NDArray


################################################################################

print(f"Device selected: {device}")


################################################################################

# Generate the dataset
n_points = 10_000

data = np.concatenate(
    [
        2.0 + 0.5 * np.random.randn(n_points // 2),
        -2.0 + 1.0 * np.random.randn(n_points // 2),
    ]
).reshape(-1, 1)

# Prepare the dataloader
batch_size = 2048
dataset = torch.from_numpy(data).float()
dataset = dataset.to(device)
dataset = TensorDataset(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Train


class GMMPath(RectifiedPath):

    def __init__(
        self,
        m: NDArray[np.float64],
        s: NDArray[np.float64],
        p: NDArray[np.float64],
        sig_min: float = 0.001,
        eps=1e-5,
    ) -> None:
        assert np.all(s > 0)
        assert np.all(p > 0)

        super().__init__()
        self.sig_min = sig_min
        self.eps = eps
        self.m = m
        self.s = s
        self.p = p / np.sum(p)

    def p_0(self, shape) -> Tensor:
        choices = np.random.choice(np.arange(len(self.p)), size=shape, p=self.p)
        res = self.m[choices] + self.s[choices] * np.random.randn(shape)
        return res


pd = RectifiedPath()
net = Net(1, 1, [512] * 5, 10).to(device)
v_t = VectorField(net)

# configure optimizer
optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)
n_epochs = 200

losses = np.zeros((n_epochs))

for epoch in tqdm(range(n_epochs), ncols=88):
    for batch in dataloader:
        x_1 = batch[0]

        # Epoch step
        optimizer.zero_grad()
        loss = fm_loss(v_t, pd, x_1)
        loss.backward()
        optimizer.step()

        # Record loss
        losses[epoch] = loss.item()

# Display the learning curve
plt.plot(np.arange(len(losses)), losses)
plt.xlabel("Epoch")
plt.ylabel("Flow Matching Loss")
plt.savefig("outputs/1d_learning_curve.png", dpi=300)
plt.close()

################################################################################

# Generate samples using the learned vector field
n_samples = 10_000
with torch.no_grad():
    x_0 = pd.p_0((n_samples, 1)).to(device=device)
    x_1_hat = v_t.decode(x_0)

x_1_hat = x_1_hat.cpu().numpy()

# Display the generated data
plt.hist(data, bins=128, alpha=0.5, label="Data")
plt.hist(x_1_hat, bins=128, alpha=0.5, label="Generated")
plt.tight_layout()
plt.legend()
plt.savefig("outputs/1d_sample_comparison.png", dpi=300)
plt.close()

################################################################################

from matplotlib.collections import PathCollection
from matplotlib.path import Path


def to_path_code(path):
    assert len(path) >= 2
    codes = [Path.MOVETO] + [Path.LINETO] * (len(path) - 1)
    return codes


# Sampling
n_samples = 200
n_steps = 200
t_steps = torch.linspace(0, 1, n_steps, device=device)

x_t = []
with torch.no_grad():
    x_t.append(pd.p_0((n_samples, 1)).to(device=device))
    for i in range(1, n_steps):
        x_t.append(v_t.decode_t0_t1(x_t[-1], t_steps[i - 1], t_steps[i]))

# Processing
x_t_numpy = np.zeros((n_steps, n_samples, 2))
for i in range(n_steps):
    x_t_numpy[i] = np.hstack([np.full((n_samples, 1), t_steps[i].item()), x_t[i].detach().cpu().numpy()])

paths = x_t_numpy.transpose(1, 0, 2)

# Drawing
fig, ax = plt.subplots(figsize=(10, 6))

for path in paths:
    ax.plot(path[:, 0], path[:, 1], color="blue", alpha=0.2)

plt.savefig("outputs/1d_sample_gen_trajectory.png", dpi=300)
plt.close()
