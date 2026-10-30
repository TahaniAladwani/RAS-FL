"""Local objectives. FedDrop/OFA/SLEX are the paper's explicitly styled adaptations."""
import copy
import random
import torch
from torch.nn import functional as F
from models import LEVELS, set_level

METHODS = ('fedavg', 'fedprox', 'feddrop_style', 'slimmable', 'ofa_style', 'slex_style', 'rasfl')


def distillation(student, teacher, temperature):
    return F.kl_div(F.log_softmax(student / temperature, dim=1),
                    F.softmax(teacher / temperature, dim=1), reduction='batchmean') * temperature ** 2


def estimate_importance(model, structure, loader, batches, device):
    # Clone protects both weights AND BN buffers of the round-start checkpoint.
    probe = copy.deepcopy(model).to(device).train()
    set_level(probe, 0)
    scores, used = structure.zeros(), 0
    for x, y in loader:
        # An adaptation client can have one example; BN cannot estimate variance
        # on a singleton final feature map. Use stored BN stats for that batch.
        for module in probe.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train(len(y) > 1)
        probe.zero_grad(set_to_none=True)
        F.cross_entropy(probe(x.to(device)), y.to(device)).backward()
        batch_scores = structure.scores(probe)
        for group in scores:
            scores[group] += batch_scores[group]
        used += 1
        if used >= batches:
            break
    if not used:
        raise ValueError('Cannot estimate importance on an empty loader')
    return {g: v / used for g, v in scores.items()}


def ema(previous, current, beta):
    return {g: beta * previous[g] + (1 - beta) * current[g] for g in previous}


def ras_levels(round_number, warmup, configured=None):
    if round_number <= warmup:
        return []
    if configured is not None:
        if not configured or len(set(configured)) != len(configured) or any(p not in LEVELS[1:] for p in configured):
            raise ValueError("Invalid training_removal_levels")
        return list(configured)
    if round_number <= 30:
        return [5, 10, 15, 20]
    if round_number <= 50:
        return [5, 10, 15, 20, 25, 30]
    return list(LEVELS[1:])


def proximal_penalty(model, reference):
    return sum((p - reference[n]).square().sum() for n, p in model.named_parameters()) / 2


def local_train(model, structure, data_loader, method, cfg, round_number, scores, seed, device):
    if method not in METHODS:
        raise ValueError(method)
    model.train()
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    reference = {n: p.detach().clone() for n, p in model.named_parameters()} if method == 'fedprox' else None
    teacher = copy.deepcopy(model).eval() if method == 'rasfl' and cfg['kd_alpha'] > 0 else None
    if teacher is not None:
        teacher.requires_grad_(False)
        set_level(teacher, 0)
    passes, updates, loss_total, loss_calls = 0, 0, 0., 0
    prefix = {p: structure.masks(p, kind='prefix') for p in LEVELS}
    ras = ({p: structure.masks(p, kind='prefix') for p in LEVELS}
           if cfg.get('ordering', 'importance') == 'prefix' else
           {p: structure.masks(p, scores) for p in LEVELS}) if method == 'rasfl' else None

    def backward(x, y, level, masks, scale=1., kd=None, all_exits=False):
        nonlocal passes, loss_total, loss_calls
        set_level(model, level)
        with structure.applied(model, masks, mask_gradients=True):
            logits = model.forward_all(x) if all_exits else [model(x)]
            if all_exits:
                weights = [.25, .35, .45, 1.]
                if len(logits) != len(weights):
                    raise ValueError('SLEX requires three intermediate exits and one final head')
                loss = sum(w * F.cross_entropy(z, y) for w, z in zip(weights, logits))
            else:
                loss = F.cross_entropy(logits[0], y)
            if kd is not None:
                alpha = cfg['kd_alpha']
                loss = (1 - alpha) * loss + alpha * distillation(logits[-1], kd, cfg['temperature'])
            if reference is not None:
                loss = loss + cfg['prox_mu'] * proximal_penalty(model, reference)
            (loss * scale).backward()
            loss_total += float(loss.detach())
            loss_calls += 1
        passes += 1

    for _ in range(cfg['local_epochs']):
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if method in ('fedavg', 'fedprox'):
                backward(x, y, 0, prefix[0])
            elif method == 'rasfl':
                available = ras_levels(round_number, cfg['warmup_rounds'], cfg.get('training_removal_levels'))
                if available and rng.random() < cfg['mask_probability']:
                    count = min(cfg.get('masked_levels_per_batch', 1), len(available))
                    selected_levels = rng.sample(available, count)
                else:
                    selected_levels = [0]
                with torch.no_grad():
                    target = teacher(x) if selected_levels != [0] and cfg['kd_alpha'] > 0 else None
                accumulated = {}
                for p in selected_levels:
                    optimizer.zero_grad(set_to_none=True)
                    backward(x, y, p, ras[p], kd=target, scale=1 / len(selected_levels))
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            accumulated[name] = accumulated.get(name, 0) + parameter.grad.detach().clone()
                optimizer.zero_grad(set_to_none=True)
                for name, parameter in model.named_parameters():
                    if name in accumulated:
                        parameter.grad = accumulated[name]
            elif method == 'feddrop_style':
                p = rng.choice(LEVELS[1:])
                backward(x, y, p, structure.masks(p, kind='random', generator=generator))
            elif method == 'slimmable':
                # Enumerate fixed widths, sharing one SGD step; separate BN at each width.
                # Save per-pass gradients before masking: later masks must not erase earlier gradients.
                accumulated = {}
                for p in LEVELS:
                    optimizer.zero_grad(set_to_none=True)
                    backward(x, y, p, prefix[p], scale=1 / len(LEVELS))
                    for n, param in model.named_parameters():
                        if param.grad is not None:
                            accumulated[n] = accumulated.get(n, 0) + param.grad.detach().clone()
                optimizer.zero_grad(set_to_none=True)
                for n, param in model.named_parameters():
                    if n in accumulated:
                        param.grad = accumulated[n]
            elif method == 'ofa_style':
                # Supplied supplement's full step followed by one random nested-width step.
                backward(x, y, 0, prefix[0])
                optimizer.step()
                updates += 1
                optimizer.zero_grad(set_to_none=True)
                p = rng.choice(LEVELS[1:])
                backward(x, y, p, prefix[p])
            elif method == 'slex_style':
                backward(x, y, 0, prefix[0], all_exits=True)
                optimizer.step()
                updates += 1
                optimizer.zero_grad(set_to_none=True)
                p = rng.choice(LEVELS[1:])
                backward(x, y, p, prefix[p], all_exits=True)
            optimizer.step()
            updates += 1
    set_level(model, 0)
    return {'student_forward_passes': passes, 'optimizer_steps': updates,
            'objective_loss_sum_over_student_passes': loss_total,
            'objective_student_pass_count': loss_calls}


class WeightedAccumulator:
    """Streaming sample-weighted FedAvg, avoiding 20 simultaneous GPU models.
    Floating buffers use the same weights. Integer BN counters use the maximum.
    """
    def __init__(self):
        self.state, self.total = None, 0

    def add(self, model, examples):
        if examples <= 0:
            raise ValueError('Client weight must be positive')
        state = {n: v.detach().cpu() for n, v in model.state_dict().items()}
        if self.state is None:
            self.state = {n: torch.zeros_like(v) for n, v in state.items()}
        for n, v in state.items():
            if v.is_floating_point():
                self.state[n].add_(v, alpha=examples)
            else:
                self.state[n] = torch.maximum(self.state[n], v)
        self.total += examples

    def finish(self):
        if not self.total:
            raise ValueError('No clients aggregated')
        return {n: v / self.total if v.is_floating_point() else v for n, v in self.state.items()}
