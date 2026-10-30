"""Validation-only selection under shared parameter budgets; standard library only."""
import argparse,csv,json,math,statistics as st
from pathlib import Path
from collections import defaultdict

def select(candidates,budget):
    feasible=[r for r in candidates if r['parameters']<=budget]
    return min(feasible,key=lambda r:(-r['validation_accuracy'],r['parameters'],r['level'])) if feasible else None

def collect(paths):
    groups={}
    for path in paths:
        folder=Path(path);m=json.loads((folder/'manifest.json').read_text());c=m['config']
        if m.get('synthetic_smoke',False): raise ValueError('Synthetic run rejected')
        v=json.loads((folder/'final_validation.json').read_text())['policy_accuracy_all_levels']
        clients=list(csv.DictReader((folder/'clients.csv').open()))
        by=defaultdict(list)
        for r in clients:
            if r['variant']=='global' and r['exit']=='final':by[int(r['requested_removal'])].append(r)
        key=(c['dataset'],float(c['train_alpha']),m['method'],int(m['seed']))
        if key in groups:raise ValueError('Duplicate run: '+str(key))
        candidates=[]
        for level,rows in by.items():
            if len(rows)!=c['deployment_clients'] or len({r['client'] for r in rows})!=len(rows):raise ValueError('Incomplete client coverage')
            counts={int(r['compact_path_parameters']) for r in rows}
            if len(counts)!=1:raise ValueError('Expected one global architecture per level')
            r=dict(level=level,parameters=counts.pop(),validation_accuracy=float(v[str(level)]),test_accuracy=st.mean(float(x['acc']) for x in rows),split_sha256=m['split_sha256'])
            if not all(math.isfinite(r[k]) for k in ['parameters','validation_accuracy','test_accuracy']):raise ValueError('Nonfinite metric')
            candidates.append(r)
        if not candidates:raise ValueError('No candidates')
        groups[key]=candidates
    return groups

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--budgets',required=True,help='JSON mapping dataset to budgets in parameter counts')
    p.add_argument('--output',required=True);a=p.parse_args()
    groups=collect(a.runs);budgets=json.loads(Path(a.budgets).read_text())
    splits={};seeds=defaultdict(set)
    for (ds,alpha,method,seed),rows in groups.items():
        key=(ds,alpha,seed);h=rows[0]['split_sha256']
        if key in splits and splits[key]!=h:raise ValueError('Methods have different data splits')
        splits[key]=h;seeds[ds,alpha,method].add(seed)
    if any(s!={42,123,2024} for s in seeds.values()):raise ValueError('Require all three seeds for every method/setting')
    selected=[];summary=defaultdict(list)
    for (ds,alpha,method,seed),rows in sorted(groups.items()):
        for b in budgets[ds]:
            if not math.isfinite(b) or b<=0:raise ValueError('Invalid budget')
            r=select(rows,b)
            selected.append(dict(dataset=ds,alpha=alpha,method=method,seed=seed,budget=b,feasible=r is not None,level=r['level'] if r else '',parameters=r['parameters'] if r else '',test_accuracy_pct=100*r['test_accuracy'] if r else ''))
            summary[ds,alpha,method,b].append(r)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'selected.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    records=[]
    for (ds,alpha,method,b),rows in summary.items():
        feasible=all(r is not None for r in rows)
        vals=[100*r['test_accuracy'] for r in rows] if feasible else []
        records.append(dict(dataset=ds,alpha=alpha,method=method,budget=b,feasible_seeds=sum(r is not None for r in rows),accuracy_mean=st.mean(vals) if vals else None,accuracy_sample_sd=st.stdev(vals) if vals else None))
    (out/'summary.json').write_text(json.dumps(records,indent=2))
    (out/'protocol.json').write_text(json.dumps({'budgets':budgets,'runs':a.runs,'selection':'maximum validation accuracy; ties lower parameters then level','missing':'no accuracy summary unless all three seeds feasible'},indent=2))
if __name__=='__main__':main()
