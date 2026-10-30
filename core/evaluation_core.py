"""Proposed replacement evaluation primitives. No training runs execute on import.
Metrics use an explicit global class set. Deployment examples are partitioned once,
then split per client, so adaptation and evaluation never share an example.
"""
import math
import random
from collections import Counter, defaultdict


def classification_metrics(targets, predictions, num_classes):
    targets, predictions = list(targets), list(predictions)
    if not targets or len(targets) != len(predictions):
        raise ValueError('Equal nonempty target/prediction arrays are required')
    if num_classes < 2 or any(not isinstance(x, int) or x < 0 or x >= num_classes for x in targets + predictions):
        raise ValueError('Labels must be integer indices in the declared class set')
    support, predicted, tp = Counter(targets), Counter(predictions), Counter(t for t,p in zip(targets,predictions) if t==p)
    f1 = [2*tp[c]/(support[c]+predicted[c]) if support[c]+predicted[c] else 0.0 for c in range(num_classes)]
    recalls = [tp[c]/support[c] for c in range(num_classes) if support[c]]
    return {'acc':sum(tp.values())/len(targets), 'macro_f1':sum(f1)/num_classes,
            'balanced_acc':sum(recalls)/len(recalls), 'n_examples':len(targets),
            'macro_f1_class_policy':'all_declared_classes_zero_division_0',
            'balanced_accuracy_class_policy':'classes_present_in_targets'}


def partition_deployment(labels, n_clients, alpha, adaptation_fraction, seed):
    """One Dirichlet class partition; variable client sizes; no replacement.
    Uses one client label mixture for adaptation/evaluation in expectation.
    Fails if a realized client has fewer than two examples; never silently resamples.
    Call on a held-out pool already disjoint from training/selection data.
    """
    if n_clients < 1 or alpha <= 0 or not 0 < adaptation_fraction < 1:
        raise ValueError('Invalid partition configuration')
    rng=random.Random(seed); by_class=defaultdict(list); clients=[[] for _ in range(n_clients)]
    for index,label in enumerate(labels): by_class[label].append(index)
    for label in sorted(by_class):
        weights=[rng.gammavariate(alpha,1) for _ in range(n_clients)]
        if sum(weights)==0: raise ValueError('Numerical underflow in Dirichlet draw')
        for index,client in zip(by_class[label],rng.choices(range(n_clients),weights=weights,k=len(by_class[label]))):
            clients[client].append(index)
    result=[]
    for cid,indices in enumerate(clients):
        if len(indices)<2: raise ValueError(f'Client {cid} has fewer than two samples; predeclare another partition configuration')
        rng.shuffle(indices); n=min(len(indices)-1,max(1,int(len(indices)*adaptation_fraction)))
        result.append({'client_id':cid,'adapt_indices':indices[:n],'eval_indices':indices[n:]})
    validate_partition(result,len(labels)); return result


def validate_partition(clients, n_examples):
    seen=set()
    for client in clients:
        for kind in ['adapt_indices','eval_indices']:
            indices=client[kind]
            if not indices: raise ValueError('Empty client split')
            for i in indices:
                if not isinstance(i,int) or i<0 or i>=n_examples or i in seen:
                    raise ValueError('Repeated or invalid example index')
                seen.add(i)
    if seen!=set(range(n_examples)): raise ValueError('Partition does not cover the declared pool')


def nested_removal_order(scores, caps):
    """Independent structures only. Stable ties by (score, layer name, index)."""
    if set(scores)!=set(caps): raise ValueError('Layer sets differ')
    candidates=[]
    for layer,values in scores.items():
        if not isinstance(caps[layer],int) or not 0<=caps[layer]<=len(values): raise ValueError('Invalid removal cap')
        for j,value in enumerate(values):
            if not math.isfinite(value): raise ValueError('Nonfinite importance')
            candidates.append((value,layer,j))
    counts=Counter();order=[]
    for _,layer,j in sorted(candidates):
        if counts[layer]<caps[layer]: order.append((layer,j)); counts[layer]+=1
    return order


def removal_prefix(scores,caps,budget):
    order=nested_removal_order(scores,caps)
    if not isinstance(budget,int) or not 0<=budget<=len(order): raise ValueError('Infeasible exact removal budget')
    return order[:budget]


def select_configuration(configs,budgets):
    """Return None when none fits. Accuracy estimates must be validation-only.
    Every configuration must provide every requested resource dimension.
    """
    if any(not math.isfinite(v) or v<0 for v in budgets.values()): raise ValueError('Invalid budgets')
    feasible=[]
    for c in configs:
        a=c['validation_accuracy']; costs=c['costs']
        if not math.isfinite(a) or not 0<=a<=1: raise ValueError('Invalid validation accuracy')
        if any(k not in costs for k in budgets): raise ValueError('Missing resource measurement')
        if any(not math.isfinite(costs[k]) or costs[k]<0 for k in budgets): raise ValueError('Invalid cost')
        if all(costs[k]<=v for k,v in budgets.items()): feasible.append(c)
    return min(feasible,key=lambda c:(-c['validation_accuracy'],str(c['id']))) if feasible else None
