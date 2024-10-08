# Torch
import torch
import torch.nn as nn
from torch import Tensor

# ODE Solver
from zuko.utils import odeint

# Type-hinting
from typing import overload


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
        super().__init__()
        self.net = net

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        return self.net(t, x)

    def wrapper(self, t: Tensor, x: Tensor) -> Tensor:
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x)

    def decode_t0_t1(self, x_0: Tensor, t0: float, t1: float):
        return odeint(self.wrapper, x_0, t0, t1, self.parameters())

    def encode(self, x_1: Tensor):
        return odeint(self.wrapper, x_1, 1.0, 0.0, self.parameters())

    def decode(self, x_0: Tensor):
        return odeint(self.wrapper, x_0, 0.0, 1.0, self.parameters())


####################################################################################################


class PathDistribution:
    def __init__(self) -> None: ...

    @overload
    def psi(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor: ...

    @overload
    def psi(self, t: Tensor, x_0: Tensor, x_1: Tensor, x_t: Tensor) -> Tensor: ...

    @overload
    def psi(self, *args, **kwargs) -> Tensor: ...

    def psi(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    @overload
    def d_psi(self, x_0: Tensor, x_1: Tensor) -> Tensor: ...

    @overload
    def d_psi(self, x_0: Tensor, x_1: Tensor, x_t: Tensor) -> Tensor: ...

    @overload
    def d_psi(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor: ...

    @overload
    def d_psi(self, t: Tensor, x_0: Tensor, x_1: Tensor, x_t: Tensor) -> Tensor: ...

    @overload
    def d_psi(self, *args, **kwargs) -> Tensor: ...

    def d_psi(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def p_0(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError


class RectifiedPath(PathDistribution):

    def __init__(self, sig_min: float = 0.001, eps=1e-5) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.eps = eps

    def psi(self, t: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
        """
        Construct the path between x_0 and x_1
        """
        return (1 - (1 - self.sig_min) * t) * x_0 + t * x_1

    def d_psi(self, x_0: Tensor, x_1: Tensor) -> Tensor:
        """
        Construct the derivative of the path between x_0 and x_1
        """
        return x_1 - (1 - self.sig_min) * x_0

    def p_0(self, shape) -> Tensor:
        return torch.randn(shape)


def fm_loss(v_t: VectorField, pd: PathDistribution, x_1: Tensor) -> Tensor:
    """
    Compute the flow-matching loss.
    """

    # t ~ Unif([0, 1])
    t = (torch.rand(1) + torch.arange(len(x_1)) / len(x_1)) % 1
    t = t.to(x_1.device)
    t = t[:, None].expand(x_1.shape)

    x_0 = pd.p_0(x_1.shape).to(x_1.device)
    psi = pd.psi(t=t, x_0=x_0, x_1=x_1)
    d_psi = pd.d_psi(x_0=x_0, x_1=x_1)
    v_psi = v_t(t[:, 0], psi)

    return torch.mean((v_psi - d_psi) ** 2)


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
        freq = 2 * torch.arange(self.n_frequencies, device=t.device) * torch.pi
        t = freq * t[..., None]
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
