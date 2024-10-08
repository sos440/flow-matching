from dm_model_simple import *
from torch.utils.data import TensorDataset, DataLoader

# Other tools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


################################################################################

print(f"Device selected: {device}")


################################################################################

# Generate the dataset
n_points = 600_00
x_data_0 = 2 * torch.rand(n_points, 1, device=device) - 1

# Prepare the dataloader
batch_size = 1024
dataset = TensorDataset(x_data_0)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Train
net = Net(1, 1, [512] * 5, 10).to(device)
v_t = VectorField(net)

# configure optimizer
optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-4)
n_epochs = 100

losses = np.zeros((n_epochs))
for epoch in tqdm(range(n_epochs), ncols=88):
    for batch in dataloader:
        x_data = batch[0]
        x_noise = torch.randn_like(x_data)

        # Epoch step
        optimizer.zero_grad()
        loss = dm_loss(v_t, x_data, x_noise)
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
n_samples = 10_000
with torch.no_grad():
    x_noise = torch.randn(n_samples, 1, device=device)
    x_data_pred = v_t.decode(x_noise)

x_data_0 = x_data_0.cpu().numpy()
x_data_pred = x_data_pred.cpu().numpy()

# Display the generated data
plt.hist(x_data_0, bins=128, alpha=0.5, label="Data", density=True)
plt.hist(x_data_pred, bins=128, alpha=0.5, label="Generated", density=True)
plt.tight_layout()
plt.legend()
plt.savefig("outputs/1d_sample_comparison.png", dpi=300)
plt.close()

# ################################################################################

# from matplotlib.collections import PathCollection
# from matplotlib.path import Path


# def to_path_code(path):
#     assert len(path) >= 2
#     codes = [Path.MOVETO] + [Path.LINETO] * (len(path) - 1)
#     return codes


# # Sampling
# n_samples = 500
# n_steps = 200
# t_steps = torch.linspace(0, 1, n_steps, device=device)

# x_t = []
# with torch.no_grad():
#     x_t.append(pd.p_0((n_samples, 1)).to(device=device))
#     for i in range(1, n_steps):
#         x_t.append(v_t.decode_t0_t1(x_t[-1], t_steps[i - 1], t_steps[i]))

# # Processing
# x_t_numpy = np.zeros((n_steps, n_samples, 2))
# for i in range(n_steps):
#     x_t_numpy[i] = np.hstack([np.full((n_samples, 1), t_steps[i].item()), x_t[i].detach().cpu().numpy()])

# paths = x_t_numpy.transpose(1, 0, 2)

# # Drawing
# fig, ax = plt.subplots(figsize=(10, 6))

# for path in paths:
#     ax.plot(path[:, 0], path[:, 1], color="blue", alpha=0.1)

# plt.savefig("outputs/1d_sample_gen_trajectory.png", dpi=300)
# plt.close()
