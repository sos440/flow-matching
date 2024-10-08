from dm_model_simple import *
from torch.utils.data import TensorDataset, DataLoader

# Other tools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


################################################################################

import argparse

parser = argparse.ArgumentParser(description="1D Flow Matching")
parser.add_argument("--loss", type=str, default="fm", help="Loss function to use")
parser.add_argument("--points", type=int, default=60000, help="Number of data points")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

args = parser.parse_args()

print(f"Device selected: {device}")

################################################################################


def generate_data(n_points):
    x = 2 * torch.rand(n_points, 1, device=device) - 1
    return x


def generate_noise(n_points):
    x = torch.randn(n_points, 1, device=device)
    return x


# Generate the dataset
n_points = args.points
x_data_0 = generate_data(n_points)

# Prepare the dataloader
batch_size = 1024
dataset = TensorDataset(x_data_0)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Train
net = Net(1, 1, [512] * 5, 10).to(device)
v_t = VectorField(net)

loss_fn = None
if args.loss == "fm":
    loss_fn = fm_loss
elif args.loss == "dm":
    loss_fn = dm_loss
else:
    raise ValueError(f"Unknown loss function: {args.loss}")

# configure optimizer
optimizer = torch.optim.Adam(v_t.parameters(), lr=args.lr)
n_epochs = args.epochs

losses = np.zeros((n_epochs))
for epoch in tqdm(range(n_epochs), ncols=88):
    for batch in dataloader:
        x_data = batch[0]
        x_noise = generate_noise(x_data.shape[0])

        # Epoch step
        optimizer.zero_grad()
        loss = fm_loss(v_t, x_data, x_noise)
        loss.backward()
        optimizer.step()

        losses[epoch] += x_data.shape[0] * loss.item()

    # Record loss
    losses[epoch] /= n_points

    tqdm.write(f"Epoch {epoch + 1}/{n_epochs}, Loss: {losses[epoch]:.4f}")

# Display the learning curve
plt.plot(np.arange(len(losses)), losses)
plt.xlabel("Epoch")
plt.ylabel("Flow Matching Loss")
plt.savefig("outputs/1d_learning_curve.png", dpi=300)
plt.close()

################################################################################


# Generate samples using the learned vector field
def test_dist(n_samples=10_000):
    x_data = generate_data(n_samples)
    with torch.no_grad():
        x_noise = generate_noise(n_samples)
        x_data_pred = v_t.decode(x_noise)

    x_data = x_data.cpu().numpy()
    x_data_pred = x_data_pred.cpu().numpy()

    # Display the generated data
    plt.hist(x_data, bins=128, alpha=0.5, label="Data", density=True)
    plt.hist(x_data_pred, bins=128, alpha=0.5, label="Generated", density=True)
    plt.tight_layout()
    plt.legend()
    plt.savefig("outputs/1d_sample_comparison.png", dpi=300)
    plt.close()


test_dist()

################################################################################

from matplotlib.collections import PathCollection
from matplotlib.path import Path


def to_path_code(path):
    assert len(path) >= 2
    codes = [Path.MOVETO] + [Path.LINETO] * (len(path) - 1)
    return codes


# Generate trajectories using the learned vector field
def test_trajectories(n_samples=500, n_steps=200):
    t = torch.linspace(0, 1, n_steps, device=device)

    x_t = torch.zeros(n_steps, n_samples, 1, device=device)
    with torch.no_grad():
        x_t[0] = generate_data(n_samples)
        for i in range(1, n_steps):
            x_t[i] = v_t.odeint(x_t[i - 1], t[i - 1], t[i])

    # Processing
    x_t_numpy = np.zeros((n_steps, n_samples, 2))
    for i in range(n_steps):
        x_t_numpy[i] = np.hstack([t[i].item() * np.ones(n_samples, 1), x_t[i].cpu().numpy()])

    paths = x_t_numpy.transpose(1, 0, 2)

    # Drawing
    fig, ax = plt.subplots(figsize=(10, 6))

    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color="blue", alpha=0.1)

    plt.savefig("outputs/1d_sample_gen_trajectory.png", dpi=300)
    plt.close()


test_dist()
