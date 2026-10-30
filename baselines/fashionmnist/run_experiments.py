"""Run the two alpha cohorts in order; never combine synthetic and real results."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import csv
import hashlib
import json
import random
import statistics
import zipfile
from pathlib import Path
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset
from data import load_data, shared_splits, loader, digest
from models import make_model, LEVELS
from early_stopping import EarlyStopping
from algorithms import FIARSE, FjORD, PartialAverage, configure, parameter_targets, local_train

ROOT = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def save_pt(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp.pt')
    torch.save(value, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(model, loaders, device, predictions=False):
    model.eval()
    rows, details = [], []
    for cid, dl in enumerate(loaders):
        correct, total = 0, 0
        for x, y in dl:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            if predictions:
                details.extend({'client': cid, 'within_client_index': total+i, 'target': int(a), 'prediction': int(b)}
                               for i, (a, b) in enumerate(zip(y, pred)))
            total += len(y)
        rows.append({'client': cid, 'correct': correct, 'total': total, 'accuracy': correct/total})
    return statistics.mean(r['accuracy'] for r in rows), rows, details


def run_one(args, cfg, alpha, seed, method, datasets, targets):
    train, clean, heldout, class_map = datasets
    cohort = args.output / f'alpha_{alpha:g}'
    cohort.mkdir(parents=True, exist_ok=True)
    split = shared_splits(cfg, seed, train.targets, heldout.targets, class_map, cohort/f'split_seed_{seed}.json')
    dest = cohort / method / f'seed_{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    base = make_model('fashionmnist')
    model = (FIARSE(base) if method == 'fiarse' else FjORD(base)).to(args.device)
    # Four clients per capacity tier. Same permuted assignments for both methods and alpha cohorts.
    tiers = [0, 10, 20, 30, 40] * 4
    random.Random(seed + 700000).shuffle(tiers)
    identity = {'method': method, 'seed': seed, 'config': cfg, 'smoke': args.smoke,
                'split_sha256': digest(split), 'capacity_removal_by_client': tiers,
                'parameter_targets': targets, 
                'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                'device': args.device,
                'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    manifest = dest/'manifest.json'
    if manifest.exists():
        if not args.resume:
            raise ValueError(f'{dest} exists; enable resume or choose a fresh output')
        if json.loads(manifest.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError('Resume identity changed. Use a fresh output directory.')
        if (dest/'results.csv').exists():
            print(f'Skip completed {dest}', flush=True)
            return
    else:
        save_json(manifest, identity)
    validation = [loader(clean, c['validation'], cfg['batch_size'], seed) for c in split['train_clients']]
    stopper = EarlyStopping(cfg['early_stopping_patience'], 0.)
    history, start = [], 1
    if (dest/'last.pt').exists():
        checkpoint = torch.load(dest/'last.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        history, start = checkpoint['history'], checkpoint['round'] + 1
        stopper = EarlyStopping(**checkpoint['stopper'])
    for round_id in range(start, start if stopper.stopped else cfg['rounds']+1):
        accumulator = PartialAverage(model)
        for cid, client in enumerate(split['train_clients']):
            local_seed = seed + round_id * 10000 + cid
            seed_all(local_seed)
            local = copy.deepcopy(model)
            dl = loader(train, client['train'], cfg['batch_size'], local_seed, True)
            support = local_train(local, dl, method, tiers[cid], targets, cfg, local_seed, args.device)
            accumulator.add(local, len(client['train']), support)
            del local
        model.load_state_dict(accumulator.finish())
        values = {}
        for removal in (0, 10, 20, 30, 40):
            configure(model, removal, targets)
            values[removal] = evaluate(model, validation, args.device)[0]
        score = .5*values[0] + .5*statistics.mean(values[r] for r in (10,20,30,40))
        improved = score > stopper.best
        stopper.update(score)
        history.append({'round': round_id, 'selection_score': score, 'validation_accuracy': values})
        if improved:
            save_pt(dest/'best.pt', {'model':model.state_dict(), 'round':round_id, 'selection_score':score})
        save_pt(dest/'last.pt', {'model':model.state_dict(), 'round':round_id, 'history':history, 'stopper':stopper.state_dict()})
        save_json(dest/'history.json', history)
        print(f'alpha={alpha} {method} seed={seed} round={round_id}/{cfg["rounds"]} val={score:.5f} stale={stopper.bad_rounds}', flush=True)
        if stopper.stopped:
            break
    best = torch.load(dest/'best.pt', map_location=args.device, weights_only=False)
    model.load_state_dict(best['model'])
    test_loaders = [loader(heldout, c['test'], cfg['batch_size'], seed) for c in split['deployment_clients']]
    rows = []
    for removal in LEVELS:
        configure(model, removal, targets)
        acc, clients, predictions = evaluate(model, test_loaders, args.device, True)
        active = model.active_parameters()
        rows.append({'dataset':'fashionmnist', 'train_alpha':alpha, 'method':method, 'seed':seed,
                     'removal_reference_pct':removal, 'equal_client_accuracy_pct':100*acc,
                     'active_parameters':active, 'target_active_parameters':targets[removal],
                     'stored_tensor_bytes':sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                     'selected_round':best['round'], 'completed_rounds':history[-1]['round'],
                     'synthetic_smoke':args.smoke})
        write_csv(dest/f'clients_removal_{removal}.csv', clients)
        write_csv(dest/f'predictions_removal_{removal}.csv', predictions)
    save_json(dest/'training_stop.json', {'patience':stopper.patience,'selected_round':best['round'],
              'completed_rounds':history[-1]['round'], 'reason':'early_stopping' if stopper.stopped else 'maximum_rounds'})
    # Written last: presence means evaluation completed.
    write_csv(dest/'results.csv', rows)


def export(args):
    rows = []
    for path in args.output.glob('alpha_*/*/seed_*/results.csv'):
        rows.extend(csv.DictReader(path.open()))
    write_csv(args.output/'all_seed_results.csv', rows)
    summaries = []
    for alpha in args.alphas:
        for method in args.methods:
            group = [r for r in rows if float(r['train_alpha']) == alpha and r['method'] == method]
            full, runtime, params = [], [], []
            for seed in args.seeds:
                seed_rows = [r for r in group if int(r['seed']) == seed]
                if len(seed_rows) != len(LEVELS):
                    raise ValueError('Incomplete seed/capacity grid; no final summary exported')
                full.append(float(next(r for r in seed_rows if int(r['removal_reference_pct'])==0)['equal_client_accuracy_pct']))
                reduced = [r for r in seed_rows if int(r['removal_reference_pct']) in (5, 10, 15, 20, 25, 30, 40)]
                runtime.append(statistics.mean(float(r['equal_client_accuracy_pct']) for r in reduced))
                params.append(statistics.mean(float(r['active_parameters']) for r in reduced))
            summaries.append({'train_alpha':alpha, 'method':method, 'seeds':len(full),
              'full_accuracy_mean':statistics.mean(full),'full_accuracy_sd':statistics.stdev(full) if len(full)>1 else '',
              'capacity_average_accuracy_mean':statistics.mean(runtime),
              'capacity_average_accuracy_sd':statistics.stdev(runtime) if len(runtime)>1 else '',
              'active_parameters_mean':statistics.mean(params),
              'active_parameters_sd':statistics.stdev(params) if len(params)>1 else ''})
    write_csv(args.output/'summary.csv', summaries)
    archive = args.output/('SMOKE_RESULTS.zip' if args.smoke else 'PAPER_RESULTS_FIARSE_FjORD.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in args.output.rglob('*'):
            if path.is_file() and path.suffix in ('.csv','.json'):
                z.write(path, path.relative_to(args.output))
        for name in ('METHOD_NOTES.md','base_config.json'):
            z.write(ROOT/name, name)
    print(f'Results: {archive}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'fashionmnist_fiarse_fjord')
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    parser.add_argument('--alphas', nargs='+', type=float, default=[.1,.4])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,123,2024])
    parser.add_argument('--methods', nargs='+', choices=['fiarse','fjord'], default=['fiarse','fjord'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.rounds < 1 or args.patience < 1 or any(a<=0 for a in args.alphas):
        parser.error('Positive rounds, patience and alpha required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Choose --device cpu explicitly if intended.')
    torch.set_num_threads(args.threads)
    cfg = json.loads((ROOT/'base_config.json').read_text())
    cfg.update(rounds=2 if args.smoke else args.rounds, early_stopping_patience=args.patience)
    if args.smoke:
        args.output = args.output / 'SMOKE_SYNTHETIC'
    args.output.mkdir(parents=True, exist_ok=True)
    targets = parameter_targets()
    if args.smoke:
        seed_all(917)
        train = TensorDataset(torch.randn(1000,1,28,28), torch.arange(1000)%10)
        heldout = TensorDataset(torch.randn(400,1,28,28), torch.arange(400)%10)
        train.targets, heldout.targets = train.tensors[1], heldout.tensors[1]
        datasets = train, train, heldout, {str(i):i for i in range(10)}
    else:
        datasets = load_data(cfg, args.data_root, args.download)
    print(f'Planned {len(args.alphas)*len(args.methods)*len(args.seeds)} training runs; smoke={args.smoke}', flush=True)
    for alpha in args.alphas:
        cfg['train_alpha'] = alpha
        for method in args.methods:
            for seed in args.seeds:
                run_one(args,cfg,alpha,seed,method,datasets,targets)
    export(args)

if __name__ == '__main__':
    main()
