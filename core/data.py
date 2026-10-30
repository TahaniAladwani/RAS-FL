"""Persisted, shared train/validation and unseen-client adaptation/test splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms as T


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class TinyValidation(Dataset):
    def __init__(self, root, class_to_idx, transform):
        root = Path(root)
        rows = [line.split() for line in (root / 'val_annotations.txt').read_text().splitlines()]
        self.samples = [(root / 'images' / r[0], class_to_idx[r[1]]) for r in rows]
        self.targets = [label for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as image:
            return self.transform(image.convert('RGB')), label


def load_data(cfg, root, download=False):
    name = cfg['dataset']
    if name == 'fashionmnist':
        normal = T.Normalize((.2860,), (.3530,))
        train_tf = eval_tf = T.Compose([T.ToTensor(), normal])
        cls = datasets.FashionMNIST
    elif name in ('cifar10', 'cifar100'):
        mean, std = ((.4914, .4822, .4465), (.2470, .2435, .2616)) if name == 'cifar10' else ((.5071, .4867, .4408), (.2675, .2565, .2761))
        eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), eval_tf])
        cls = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
    else:
        root = Path(root) / 'tiny-imagenet-200'
        eval_tf = T.Compose([T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
        train_tf = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), eval_tf])
        train = datasets.ImageFolder(root / 'train', transform=train_tf)
        if len(train.classes) != 200:
            raise ValueError('Tiny-ImageNet must have exactly 200 classes')
        clean = datasets.ImageFolder(root / 'train', transform=eval_tf)
        heldout = TinyValidation(root / 'val', train.class_to_idx, eval_tf)
        return train, clean, heldout, train.class_to_idx
    train = cls(root=root, train=True, download=download, transform=train_tf)
    clean = cls(root=root, train=True, download=download, transform=eval_tf)
    heldout = cls(root=root, train=False, download=download, transform=eval_tf)
    return train, clean, heldout, {str(i): i for i in range(cfg['classes'])}


def partition(labels, clients, alpha, rng, minimum=2, attempts=1000):
    """Class-wise Dirichlet-multinomial allocation, conditioned on minimum size.
    Bounded rejection and its accepted attempt number are recorded, never hidden.
    """
    labels = np.asarray(labels)
    if alpha <= 0 or clients < 1 or len(labels) < clients * minimum:
        raise ValueError('Invalid or impossible partition')
    for attempt in range(1, attempts + 1):
        out = [[] for _ in range(clients)]
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            for bucket, part in zip(out, np.split(indices, np.cumsum(counts)[:-1])):
                bucket.extend(part.tolist())
        if min(map(len, out)) >= minimum:
            for bucket in out:
                rng.shuffle(bucket)
            return out, attempt
    raise ValueError('Partition failed the declared minimum-size condition')


def validate_splits(split, ntrain, ntest):
    def check(parts, length):
        flat = [i for part in parts for i in part]
        if any(not part for part in parts) or sorted(flat) != list(range(length)):
            raise ValueError('Splits must be nonempty, disjoint and cover their source pool')
    check([s[k] for s in split['train_clients'] for k in ('train', 'validation')], ntrain)
    check([s[k] for s in split['deployment_clients'] for k in ('adapt', 'test')], ntest)


def shared_splits(cfg, seed, train_labels, test_labels, class_map, path):
    path = Path(path)
    identity = {'dataset': cfg['dataset'], 'seed': seed, 'training_clients': cfg['training_clients'],
                'deployment_clients': cfg['deployment_clients'], 'train_alpha': cfg['train_alpha'],
                'deployment_alpha': cfg['deployment_alpha'], 'validation_fraction': cfg['validation_fraction'],
                'adaptation_fraction': cfg['adaptation_fraction'], 'class_map': class_map,
                'train_labels_sha256': digest(list(map(int, train_labels))),
                'test_labels_sha256': digest(list(map(int, test_labels)))}
    if path.exists():
        result = json.loads(path.read_text())
        if result['identity'] != identity:
            raise ValueError('Existing split identity differs; choose a fresh output directory')
    else:
        rng = np.random.default_rng(seed)
        training, ta = partition(train_labels, cfg['training_clients'], cfg['train_alpha'], rng, minimum=3)
        deploy, da = partition(test_labels, cfg['deployment_clients'], cfg['deployment_alpha'], rng)
        result = {'identity': identity, 'accepted_partition_attempt': {'train': ta, 'deployment': da},
                  'train_clients': [], 'deployment_clients': []}
        for indices in training:
            nv = min(len(indices) - 1, max(1, round(len(indices) * cfg['validation_fraction'])))
            result['train_clients'].append({'train': indices[nv:], 'validation': indices[:nv]})
        for indices in deploy:
            na = min(len(indices) - 1, max(1, round(len(indices) * cfg['adaptation_fraction'])))
            result['deployment_clients'].append({'adapt': indices[:na], 'test': indices[na:]})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    validate_splits(result, len(train_labels), len(test_labels))
    return result


def loader(dataset, indices, batch_size, seed, shuffle=False):
    if shuffle:
        if len(indices) < 2 or batch_size < 2:
            raise ValueError('Training/importance loader requires at least two examples and batch size >= 2')
        # Merge a singleton tail into its predecessor; do not drop any training example.
        class Batches:
            def __init__(self):
                self.generator = torch.Generator().manual_seed(seed)

            def __iter__(self):
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
                if len(chunks) > 1 and len(chunks[-1]) == 1:
                    chunks[-2].extend(chunks.pop())
                return iter(chunks)

            def __len__(self):
                n = (len(indices) + batch_size - 1) // batch_size
                return n - int(n > 1 and len(indices) % batch_size == 1)
        return DataLoader(Subset(dataset, indices), batch_sampler=Batches(), num_workers=0)
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=False)
