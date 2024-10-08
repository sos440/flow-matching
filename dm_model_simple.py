# Torch
import torch
import torch.nn as nn
from torch import Tensor

# ODE Solver
from zuko.utils import odeint


# Choose device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


####################################################################################################


class VectorField(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        """
        Construct a vector field from a neural network.
        This is essentially a wrapper around the neural network that allows to solve ODEs.

        Args:
            `net` (`nn.Module`): The neural network representing the vector field. It must take two inputs:

                - `t` (`Tensor`): The time. It is either a scalar or a tensor of shape `(batch_size)`.
                - `x` (`Tensor`): The state. It is a tensor of shape `(batch_size, state_dim)`.

            and output the vector field at the given time and state. It must return a tensor of shape `(batch_size, state_dim)`.

        Returns:
            None
        """
        super().__init__()
        self.net = net

    def forward(self, t: float | Tensor, x: Tensor) -> Tensor:
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device=x.device)
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        return self.net(t, x)

    def odeint(self, x_0: Tensor, t_0: float, t_1: float) -> Tensor:
        """
        Solve the ODE given the initial state `x0` at time `t0` and the final time `t1`.
        """
        return odeint(self.forward, x_0, t_0, t_1, self.parameters())

    def encode(self, x_data: Tensor) -> Tensor:
        """
        Return the noised samples given the data samples.
        """
        return self.odeint(x_data, 0.0, 1.0)

    def decode(self, x_noise: Tensor):
        """
        Return the data samples given the noised samples.
        """
        return self.odeint(x_noise, 1.0, 0.0)


####################################################################################################


def dm_loss(v: VectorField, x_data: Tensor, x_noise: Tensor) -> Tensor:
    """
    Compute the direct-matching loss.
    """

    assert x_data.shape == x_noise.shape, "The data and noise tensors must have the same shape."
    assert x_data.device == x_noise.device, "The data and noise tensors must be on the same device."

    t = torch.rand(x_data.shape[0], device=x_data.device)[:, None]
    s = torch.rand(x_data.shape[0], device=x_data.device)[:, None]

    loss = torch.sum((1 - t) * (v(t, x_data) ** 2))
    loss += torch.sum(t * (v(t, x_noise) ** 2))
    loss += -2 * torch.sum((x_noise - x_data) * v(t, (1 - s) * x_data + s * x_noise))
    loss /= x_data.shape[0]

    return loss


def fm_loss(v: VectorField, x_data: Tensor, x_noise: Tensor) -> Tensor:
    """
    Compute the flow-matching loss.
    """

    assert x_data.shape == x_noise.shape, "The data and noise tensors must have the same shape."
    assert x_data.device == x_noise.device, "The data and noise tensors must be on the same device."

    # Time
    t = torch.rand(x_data.shape[0], device=x_data.device)[:, None]

    loss = torch.sum((v(t, (1 - t) * x_data + t * x_noise) - (x_noise - x_data)) ** 2) / x_data.shape[0]

    return loss


def mixed_loss(v: VectorField, x_data: Tensor, x_noise: Tensor) -> Tensor:
    """
    Compute the flow-matching loss.
    """

    assert x_data.shape == x_noise.shape, "The data and noise tensors must have the same shape."
    assert x_data.device == x_noise.device, "The data and noise tensors must be on the same device."

    loss = fm_loss(v, x_data, x_noise) + 0.1 * dm_loss(v, x_data, x_noise)

    return loss


####################################################################################################


class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, h_dims: list[int], n_frequencies: int) -> None:
        super().__init__()

        ins = [in_dim + 2 * n_frequencies] + h_dims
        outs = h_dims + [out_dim]
        self.n_frequencies = n_frequencies

        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)]
        )
        self.top = nn.Sequential(nn.Linear(out_dim, out_dim))

    def time_encoder(self, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t[..., None]
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t = freq * t
        return torch.cat((t.cos(), t.sin()), dim=-1)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        # Extend the input with the time encoding
        t = self.time_encoder(t)
        x = torch.cat((x, t), dim=-1)

        # Pass through the layers
        for l in self.layers:
            x = l(x)
        x = self.top(x)

        return x
