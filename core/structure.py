"""Channel dependency discovery, reversible masking and physical compaction.

Budget unit: one independent dependency-group coordinate. Residual branches and
depthwise input/output coordinates are tied. This explicit extension is required
for the named backbones; it is not the original independent-layer theorem.
"""
import copy
import math
import operator
from contextlib import contextmanager
import torch
from torch import nn, fx
from models import SwitchBN, set_level


class Tracer(fx.Tracer):
    def is_leaf_module(self, module, name):
        return isinstance(module, SwitchBN) or super().is_leaf_module(module, name)


class Structure:
    def __init__(self, model, minimum=.5, first_minimum=.7):
        graph = Tracer().trace(model)
        parent, sizes, node_group, specs, norm, stages = {}, {}, {}, {}, {}, {}
        protected = set()
        first = None

        def new(key, size):
            parent[key], sizes[key] = key, size
            return key

        def root(key):
            while parent[key] != key:
                key = parent[key]
            return key

        def union(a, b):
            a, b = root(a), root(b)
            if sizes[a] != sizes[b]:
                raise ValueError('Incompatible residual channel counts')
            parent[max(a, b)] = min(a, b)

        for node in graph.nodes:
            if node.op == 'placeholder':
                node_group[node] = None
                continue
            if node.op == 'output':
                protected.add(node_group[node.args[0]])
                continue
            inputs = [a for a in node.all_input_nodes if a in node_group]
            incoming = node_group[inputs[0]] if inputs else None
            if node.op == 'call_module':
                module = model.get_submodule(node.target)
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    out = module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
                    outgoing = new(node.target, out)
                    if isinstance(module, nn.Conv2d):
                        if first is None:
                            first = outgoing
                        if module.groups != 1:
                            if module.groups != module.in_channels or out != module.in_channels or incoming is None:
                                raise NotImplementedError('Only depthwise groups with multiplier 1 supported')
                            union(incoming, outgoing)
                    specs[node.target] = (incoming, outgoing)
                elif isinstance(module, nn.BatchNorm2d):
                    outgoing = incoming
                    norm[node.target] = outgoing
                elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Dropout, nn.Dropout2d,
                                         nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Identity)):
                    outgoing = incoming
                else:
                    raise NotImplementedError(f'Unrecognized module {node.target}: {type(module)}')
                if node.target.startswith('stages.'):
                    stages[int(node.target.split('.')[1])] = outgoing
            elif node.op == 'call_function' and node.target in (operator.add, torch.add):
                a, b = (node_group[v] for v in inputs[:2])
                union(a, b)
                outgoing = a
            elif node.op == 'call_function' and node.target == torch.flatten:
                outgoing = incoming  # Every head uses spatial global average pooling first.
            else:
                raise NotImplementedError(f'Unsupported graph operation: {node.op} {node.target}')
            node_group[node] = outgoing
        self.specs = {n: (root(a) if a else None, root(b)) for n, (a, b) in specs.items()}
        self.norm = {n: root(g) for n, g in norm.items()}
        self.sizes = {root(k): sizes[k] for k in sizes}
        self.protected = {root(g) for g in protected}
        for i, module in enumerate(model.exits):
            key = f'exits.{i}'
            self.sizes[key] = module.out_features
            self.protected.add(key)
            self.specs[key] = (root(stages[i]), key)
        self.caps = {g: 0 if g in self.protected else n - math.ceil(n * (first_minimum if g == root(first) else minimum))
                     for g, n in self.sizes.items()}
        self.eligible = sorted(g for g in self.sizes if g not in self.protected)
        self.total = sum(self.sizes[g] for g in self.eligible)

    def zeros(self):
        return {g: torch.zeros(self.sizes[g]) for g in self.eligible}

    def scores(self, model, gradient=True):
        result = self.zeros()
        for name, (_, group) in self.specs.items():
            if group not in result:
                continue
            weight = model.get_submodule(name).weight
            if gradient and weight.grad is None:
                continue
            value = (weight * weight.grad if gradient else weight).detach().abs()
            result[group] += value.flatten(1).sum(1).cpu()
        return result

    def masks(self, percent, scores=None, kind='importance', generator=None):
        if not 0 <= percent <= 100:
            raise ValueError('Removal percentage out of range')
        masks = {g: torch.ones(n, dtype=torch.bool) for g, n in self.sizes.items()}
        if kind == 'prefix':
            # Width baselines have per-group rounding, not the global RAS budget.
            for g in self.eligible:
                remove = min(self.caps[g], round(self.sizes[g] * percent / 100))
                if remove:
                    masks[g][-remove:] = False
            return masks
        budget = round(percent * self.total / 100)
        if budget > sum(self.caps.values()):
            raise ValueError('Infeasible exact dependency-group removal budget')
        if kind == 'random':
            scores = {g: torch.rand(self.sizes[g], generator=generator) for g in self.eligible}
        if scores is None or set(scores) != set(self.eligible):
            raise ValueError('Scores do not cover the eligible groups')
        candidates = []
        for g in self.eligible:
            if scores[g].numel() != self.sizes[g] or not torch.isfinite(scores[g]).all():
                raise ValueError('Invalid importance scores')
            candidates.extend((float(s), g, j) for j, s in enumerate(scores[g]))
        counts = dict.fromkeys(self.eligible, 0)
        removed = 0
        for _, g, j in sorted(candidates):
            if removed == budget:
                break
            if counts[g] < self.caps[g]:
                masks[g][j] = False
                counts[g] += 1
                removed += 1
        return masks

    def parameter_masks(self, model, masks):
        result = {}
        for name, (inp, out) in self.specs.items():
            module = model.get_submodule(name)
            active_out = masks[out].to(module.weight.device)
            active_in = masks[inp].to(module.weight.device) if inp else torch.ones(module.weight.shape[1], device=module.weight.device, dtype=torch.bool)
            if isinstance(module, nn.Conv2d):
                mask = active_out[:, None, None, None]
                if module.groups == 1:
                    mask = mask & active_in[None, :, None, None]
            else:
                mask = active_out[:, None] & active_in[None, :]
            result[name + '.weight'] = mask
            if module.bias is not None:
                result[name + '.bias'] = active_out
        for name, g in self.norm.items():
            module = model.get_submodule(name)
            for suffix, bn in [('', module)] + ([(f'.banks.{p}', bank) for p, bank in module.banks.items()] if isinstance(module, SwitchBN) else []):
                result[name + suffix + '.weight'] = masks[g].to(bn.weight.device)
                result[name + suffix + '.bias'] = masks[g].to(bn.bias.device)
        return result

    @contextmanager
    def applied(self, model, masks, mask_gradients=False):
        """Restore weights BEFORE optimizer.step; momentum/decay remain standard SGD."""
        params = dict(model.named_parameters())
        pm = self.parameter_masks(model, masks)
        backup = {n: params[n].detach().clone() for n in pm}
        try:
            with torch.no_grad():
                for n, m in pm.items():
                    params[n].mul_(m)
            yield
        finally:
            with torch.no_grad():
                for n, value in backup.items():
                    params[n].copy_(value)
                    if mask_gradients and params[n].grad is not None:
                        params[n].grad.mul_(pm[n])

    def compact(self, model, masks, level=0):
        """Copy and slice every linked input/output, residual and depthwise channel."""
        compact = copy.deepcopy(model).eval()
        for name, (inp, out) in self.specs.items():
            old = model.get_submodule(name)
            oi = masks[out].nonzero().flatten().to(old.weight.device)
            ni = old.in_channels if isinstance(old, nn.Conv2d) else old.in_features
            ii = masks[inp].nonzero().flatten().to(old.weight.device) if inp else torch.arange(ni, device=old.weight.device)
            if isinstance(old, nn.Conv2d):
                depthwise = old.groups != 1
                new = nn.Conv2d(len(ii), len(oi), old.kernel_size, old.stride, old.padding,
                                old.dilation, len(ii) if depthwise else 1, old.bias is not None,
                                old.padding_mode).to(device=old.weight.device, dtype=old.weight.dtype)
                w = old.weight[oi] if depthwise else old.weight[oi][:, ii]
            else:
                new = nn.Linear(len(ii), len(oi), old.bias is not None).to(device=old.weight.device, dtype=old.weight.dtype)
                w = old.weight[oi][:, ii]
            with torch.no_grad():
                new.weight.copy_(w)
                if old.bias is not None:
                    new.bias.copy_(old.bias[oi])
            parent, _, key = name.rpartition('.')
            setattr(compact.get_submodule(parent), key, new)
        for name, group in self.norm.items():
            old = model.get_submodule(name)
            if isinstance(old, SwitchBN) and level:
                old = old.banks[str(level)]
            idx = masks[group].nonzero().flatten().to(old.weight.device)
            new = nn.BatchNorm2d(len(idx), eps=old.eps, momentum=old.momentum).to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                for key in ('weight', 'bias', 'running_mean', 'running_var'):
                    getattr(new, key).copy_(getattr(old, key)[idx])
                new.num_batches_tracked.copy_(old.num_batches_tracked)
            parent, _, key = name.rpartition('.')
            setattr(compact.get_submodule(parent), key, new)
        return compact.eval()
