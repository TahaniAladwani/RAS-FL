"""Extract an importance roadmap from an already selected FedAvg checkpoint."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'core'))
from run import (torch, seed_all, state_hash, make_model, Structure, loader,
                 load_data, digest, estimate_importance, ema, roadmap,
                 validation_score, evaluate_deployment, LEVELS)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--download', action='store_true')
    a=p.parse_args()
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if saved['method']!='fedavg': raise ValueError('Requires a selected FedAvg deployment.pt')
    cfg=saved['config']; seed=saved['seed']; device=torch.device(a.device)
    split=json.loads(Path(a.split).read_text())
    if digest(split)!=saved['split_sha256']: raise ValueError('Checkpoint/split mismatch')
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    seed_all(seed)
    train,clean,heldout,_=load_data(cfg,a.data_root,a.download)
    model=make_model(cfg['dataset']).to(device); model.load_state_dict(saved['model'])
    before=state_hash(model.state_dict())
    structure=Structure(model,cfg['min_active_ratio'],cfg['first_min_active_ratio'])
    scores=structure.zeros()
    # Same fixed-checkpoint stabilization routine as RAS-FL, initialized at zero:
    # no training-time importance history exists for ordinary FedAvg.
    for step in range(cfg['stabilization_passes']):
        combined=structure.zeros(); total=0
        for cid,indices in enumerate(split['train_clients']):
            local_seed=seed+600000+step*10000+cid; seed_all(local_seed)
            dl=loader(train,indices['train'],cfg['batch_size'],local_seed,shuffle=True)
            local=estimate_importance(model,structure,dl,cfg['importance_batches'],device)
            n=len(indices['train']);total+=n
            for g in combined: combined[g]+=n*local[g]
        scores=ema(scores,{g:v/total for g,v in combined.items()},cfg['importance_beta'])
    if cfg['stabilization_passes']<1: raise ValueError('At least one importance pass required')
    masks=roadmap(structure,'rasfl',scores,{})
    validation=[loader(clean,c['validation'],cfg['batch_size'],seed) for c in split['train_clients']]
    _,values=validation_score(model,structure,masks,validation,cfg['classes'],device,True,levels=list(LEVELS))
    (out/'final_validation.json').write_text(json.dumps({'policy_accuracy_all_levels':values},indent=2))
    rows=evaluate_deployment(model,structure,'fedavg_posthoc',masks,scores,cfg,split,heldout,seed,device,out)
    with (out/'clients.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert before==state_hash(model.state_dict()), 'Model parameters or buffers changed'
    torch.save({**saved,'method':'fedavg_posthoc','scores':scores,'masks':masks},out/'deployment.pt')
    (out/'manifest.json').write_text(json.dumps({'config':cfg,'seed':seed,'method':'fedavg_posthoc',
        'split_sha256':digest(split),'source_checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        'weights_and_buffers_unchanged':True,'selected_round':saved['selected_round'],
        'importance_initialization':'zero; fixed-checkpoint training-client sensitivity; no optimizer steps',
        'core_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).parent/'core').glob('*.py')}},indent=2))
    print(out)
if __name__=='__main__': main()
