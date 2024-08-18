from fm_model_simple import *
from torch.utils.data import TensorDataset, DataLoader

# SK-Learn datasets
from sklearn.datasets import make_moons, make_swiss_roll, make_checkerboard
from sklearn.preprocessing import StandardScaler

# Other tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


################################################################################

print(f"Device selected: {device}")


# Generate the checkerboard dataset
def generate_checkerboard(n_points: int, n_squares: int = 4, range_min=-1, range_max=1):
    square_size = (range_max - range_min) / n_squares

    position_map = np.array(
        [
            (range_min + i * square_size, range_min + j * square_size)
            for i in range(0, n_squares)
            for j in range(i % 2, n_squares, 2)
        ]
    )

    points = np.random.uniform(0, square_size, (n_points, 2))
    cells = np.random.randint(0, len(position_map), n_points)
    points = points + position_map[cells]

    return points


def get_data(dataset: str, n_points: int) -> np.ndarray:
    match dataset:
        case "moons":
            data, _ = make_moons(n_points, noise=0.15)
        case "swiss":
            data, _ = make_swiss_roll(n_points, noise=0.25)
            data = data[:, [0, 2]] / 10.0
        case "chekerboard":
            data = generate_checkerboard(n_points, 4, -1.6, 1.6)
        case _:
            raise ValueError(f"Unknown dataset: {dataset}")

    return StandardScaler().fit_transform(data)


################################################################################


# Choose dataset
dataset = "chekerboard"
n_points = 100_000

data = get_data(dataset, n_points)

# Display the data
plt.hist2d(data[:, 0], data[:, 1], bins=128)
plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.5)
plt.tight_layout()
plt.gca().set_aspect("equal", adjustable="box")
plt.savefig("outputs/2d_sample_original.png", dpi=300)
plt.close()

# Prepare the dataloader
batch_size = 2048
dataset = torch.from_numpy(data).float()
dataset = dataset.to(device)
dataset = TensorDataset(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Train
pd = RectifiedPath()
net = Net(2, 2, [512] * 5, 10).to(device)
v_t = VectorField(net)

# configure optimizer
optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)
n_epochs = 2000

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
plt.savefig("outputs/2d_learning_curve.png", dpi=300)
plt.close()

################################################################################

# Generate samples using the learned vector field
n_samples = 10_000
with torch.no_grad():
    x_0 = pd.p_0((n_samples, 2)).to(device=device)
    x_1_hat = v_t.decode(x_0)

x_1_hat = x_1_hat.cpu().numpy()

# Display the generated data
plt.hist2d(x_1_hat[:, 0], x_1_hat[:, 1], bins=128)
plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.5)
plt.tight_layout()
plt.gca().set_aspect("equal", adjustable="box")
plt.savefig("outputs/2d_sample_generated.png", dpi=300)
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

with torch.no_grad():
    x_t = [torch.randn(n_samples, 2, device=device)]
    for t in range(len(t_steps) - 1):
        x_t += [v_t.decode_t0_t1(x_t[-1], t_steps[t], t_steps[t + 1])]

x_t_numpy = np.array([x.detach().cpu().numpy() for x in x_t])

# Drawing
paths = x_t_numpy.transpose(1, 0, 2)
vertices = np.array(paths).reshape(-1, 2)
path_codes = np.concatenate([to_path_code(p) for p in paths])

path_collection = PathCollection(
    [Path(vertices, path_codes)],
    linewidths=1,
    facecolors="none",
    edgecolors="black",
)

fig, ax = plt.subplots(figsize=(8, 8))
ax.add_collection(path_collection)
ax.scatter(x_t_numpy[-1, :, 0], x_t_numpy[-1, :, 1], s=10, c="red")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal", adjustable="box")
plt.savefig("outputs/2d_sample_gen_trajectory.png", dpi=300)
plt.close()