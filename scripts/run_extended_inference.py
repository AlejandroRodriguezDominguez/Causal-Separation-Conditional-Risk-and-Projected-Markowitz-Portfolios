#!/usr/bin/env python3
"""Paired fold inference for the extended empirical experiments."""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'extended_results'
RNG=np.random.default_rng(20260903)


def signflip_p(values, draws=200000):
    x=np.asarray(values,float); observed=x.mean()
    if len(x)<=18:
        means=np.array([(x*np.asarray(s)).mean() for s in itertools.product((-1,1),repeat=len(x))])
    else:
        means=(RNG.choice((-1,1),size=(draws,len(x)))*x).mean(axis=1)
    return float((1+(means<=observed).sum())/(1+len(means)))


def bootstrap_ci(values,draws=20000):
    x=np.asarray(values,float); means=RNG.choice(x,size=(draws,len(x)),replace=True).mean(1)
    return tuple(np.quantile(means,[.025,.975]))


def holm(frame):
    p=frame.p_value.to_numpy(); m=len(p); order=np.argsort(p); adjusted=np.empty(m)
    running=0.0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]); adjusted[idx]=min(running,1.0)
    frame['holm_p']=adjusted; return frame


rows=[]
multi=pd.read_csv(OUT/'multiasset_fold_results.csv')
for panel,g in multi.groupby('panel'):
    ci=bootstrap_ci(g.delta_s_f)
    rows.append({'family':'multiasset','comparison':panel,'folds':len(g),'mean_delta':g.delta_s_f.mean(),
                 'ci_low':ci[0],'ci_high':ci[1],'p_value':signflip_p(g.delta_s_f)})
    if panel == 'equities_150' and 'state_delta_vs_constant_s_f' in g:
        z=g.state_delta_vs_constant_s_f.dropna(); ci=bootstrap_ci(z)
        rows.append({'family':'state_interaction','comparison':panel,'folds':len(z),'mean_delta':z.mean(),
                     'ci_low':ci[0],'ci_high':ci[1],'p_value':signflip_p(z)})
h=pd.read_csv(OUT/'multihorizon_fold_results.csv')
for (panel,horizon),g in h.groupby(['panel','horizon_sessions']):
    ci=bootstrap_ci(g.delta_s_f)
    rows.append({'family':'multihorizon','comparison':f'{panel}_h{horizon}','folds':len(g),
                 'mean_delta':g.delta_s_f.mean(),'ci_low':ci[0],'ci_high':ci[1],
                 'p_value':signflip_p(g.delta_s_f)})
result=pd.concat([holm(g.copy()) for _,g in pd.DataFrame(rows).groupby('family')],ignore_index=True)
result.to_csv(OUT/'extended_inference.csv',index=False)
print(result.to_string(index=False))
