"""Independent Fashion-MNIST adaptations; see METHOD_NOTES.md for deviations."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from models import make_model, LEVELS


class ThresholdWeight(torch.autograd.Function):
    """FIARSE Eq. (3): fixed local threshold, changing forward mask, biased gradient."""
    @staticmethod
    def forward(ctx, weight, threshold):
        ctx.save_for_backward(weight, threshold)
        return weight * (weight.abs() >= threshold)

    @staticmethod
    def backward(ctx, gradient):
        weight, threshold = ctx.saved_tensors
        magnitude = weight.abs()
        bias = torch.nan_to_num(2 * magnitude * threshold / (magnitude + threshold).square())
        return gradient * (magnitude >= threshold) * (1 + bias), None


class FIARSE(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        self.base = base if base is not None else make_model('fashionmnist')
        self.names = {f'{name}.{key}' for name, layer in self.base.named_modules()
                      if isinstance(layer, (nn.Conv2d, nn.Linear))
                      for key, value in layer.named_parameters(recurse=False)}
        self.threshold = torch.tensor(0.)

    def set_budget(self, total_parameters):
        parameters = dict(self.base.named_parameters())
        dense = sum(p.numel() for n, p in parameters.items() if n not in self.names)
        values = torch.cat([p.detach().abs().flatten() for n, p in parameters.items() if n in self.names])
        k = max(1, min(values.numel(), int(total_parameters) - dense))
        # CUDA kthvalue is rejected under strict deterministic algorithms.
        # Compute the same order statistic on CPU, then return only its scalar.
        self.threshold = (values.new_zeros(()) if k == values.numel() else
                          values.cpu().kthvalue(values.numel() - k + 1).values.to(values.device))
        # As in the reference implementation, threshold ties are retained and actual counts recorded.
        return self.active_parameters()

    def active_parameters(self):
        return sum(int((p.detach().abs() >= self.threshold).sum()) if n in self.names else p.numel()
                   for n, p in self.base.named_parameters())

    def forward(self, x):
        params = {n: ThresholdWeight.apply(p, self.threshold.to(p.device)) if n in self.names else p
                  for n, p in self.base.named_parameters()}
        return functional_call(self.base, params, (x,))

    def support(self):
        return {'base.' + n: (p.detach().abs() >= self.threshold) if n in self.names else torch.ones_like(p, dtype=torch.bool)
                for n, p in self.base.named_parameters()}


class FjORD(nn.Module):
    """Uniform nested prefix widths, shared BN affine parameters and per-width statistics."""
    def __init__(self, base=None):
        super().__init__()
        self.base = base if base is not None else make_model('fashionmnist')
        self.banks = nn.ModuleDict()
        for i, stage in enumerate(self.base.stages):
            for removal in LEVELS:
                channels = math.ceil(stage[0].out_channels * (1 - removal / 100))
                key = f's{i}_n{channels}'
                if key not in self.banks:
                    self.banks[key] = nn.BatchNorm2d(channels, affine=False)
        self.removal = 0

    def set_removal(self, removal):
        if removal not in LEVELS:
            raise ValueError('Width must be in the declared ordered-dropout grid')
        self.removal = removal

    def support(self, removal=None):
        p = 1 - (self.removal if removal is None else removal) / 100
        out = {n: torch.zeros_like(v, dtype=torch.bool) for n, v in self.named_parameters()}
        previous = 1
        for i, stage in enumerate(self.base.stages):
            width = math.ceil(stage[0].out_channels * p)
            out[f'base.stages.{i}.0.weight'][:width, :previous] = True
            for key in ('weight', 'bias'):
                out[f'base.stages.{i}.1.{key}'][:width] = True
            previous = width
        hidden = math.ceil(128 * p)
        out['base.head.0.weight'][:hidden, :previous] = True
        out['base.head.0.bias'][:hidden] = True
        out['base.head.2.weight'][:, :hidden] = True
        out['base.head.2.bias'][:] = True
        return out

    def active_parameters(self):
        return sum(int(v.sum()) for v in self.support().values())

    def forward(self, x):
        p = 1 - self.removal / 100
        for i, stage in enumerate(self.base.stages):
            conv, bn = stage[0], stage[1]
            width = math.ceil(conv.out_channels * p)
            x = F.conv2d(x, conv.weight[:width, :x.shape[1]], padding=1)
            x = self.banks[f's{i}_n{width}'](x)
            x = x * bn.weight[:width][None, :, None, None] + bn.bias[:width][None, :, None, None]
            x = F.max_pool2d(F.relu(x), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        hidden = math.ceil(128 * p)
        first, last = self.base.head[0], self.base.head[2]
        x = F.relu(F.linear(x, first.weight[:hidden, :x.shape[1]], first.bias[:hidden]))
        return F.linear(x, last.weight[:, :hidden], last.bias)


def parameter_targets():
    # Equal active-parameter budgets between the two new methods, including BN affine/bias.
    model = FjORD()
    return {r: sum(int(v.sum()) for v in model.support(r).values()) for r in LEVELS}


def configure(model, removal, targets):
    if isinstance(model, FIARSE):
        model.set_budget(targets[removal])
    else:
        model.set_removal(removal)


class PartialAverage:
    """Data-weighted coordinate averaging; never dilute coordinates absent on a client."""
    def __init__(self, model):
        self.original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.numerator = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in self.original.items()}
        self.denominator = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in self.original.items()}

    def add(self, model, weight, support):
        for k, value in model.state_dict().items():
            mask = support.get(k, torch.zeros_like(value, dtype=torch.bool))
            self.numerator[k] += value.double() * mask * weight
            self.denominator[k] += mask * weight

    def finish(self):
        return {k: torch.where(self.denominator[k] > 0,
                              self.numerator[k] / self.denominator[k].clamp_min(1), v.double()).to(v.dtype)
                for k, v in self.original.items()}


def local_train(model, dl, method, capacity, targets, cfg, seed, device):
    import random
    rng = random.Random(seed)
    model.train()
    configure(model, capacity, targets)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    eligible = [r for r in LEVELS if r >= capacity]
    parameter_support = model.support()
    visited = set()
    for _ in range(cfg['local_epochs']):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if method == 'fjord':
                model.set_removal(rng.choice(eligible))
                visited.add(model.removal)
            step_support = model.support()
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite local loss')
            loss.backward()
            optimizer.step()
            # Keep masked coordinates fixed despite momentum/weight decay.
            with torch.no_grad():
                for n, p in model.named_parameters():
                    mask = step_support[n]
                    p.copy_(torch.where(mask, p, before[n]))
                    state = optimizer.state.get(p, {})
                    if 'momentum_buffer' in state:
                        state['momentum_buffer'].mul_(mask)
    if method == 'fiarse':
        # FIARSE reference server averages only nonzero updates, not forward-mask membership.
        support = {k: value != original[k] for k, value in model.state_dict().items()}
    else:
        support = dict(parameter_support)  # full client-capacity block, Eq. (2)
        for removal in visited:
            for i, stage in enumerate(model.base.stages):
                channels = math.ceil(stage[0].out_channels * (1-removal/100))
                prefix = f'banks.s{i}_n{channels}.'
                for k, v in model.state_dict().items():
                    if k.startswith(prefix):
                        support[k] = torch.ones_like(v, dtype=torch.bool)
    return support
