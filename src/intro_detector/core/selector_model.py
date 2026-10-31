"""Candidate times and features of the first-vocal selector (ExtraTrees ranking model)."""
import re
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from ..evidence import INPUTS

POLICY = 'intro-selector-extratrees-three-inputs-v2'
variants=INPUTS
# Common filler/hallucinated phrases, not track-specific lyrics.
bad={'thank you','to be continued',"we'll be right back",'we will be right back'}

def source(r):
    a={c['chain']:c['cue_s'] for c in r['candidates']}
    refs=[]
    for v in variants:
        for m in ('turbo','qwen'):
            s=r['asr'][v+'/'+m]
            t=s['first_word_start']
            flags=s['quality_flags'] or []
            if t is not None and 'invalid_word_timing' not in flags and 'alignment_failed' not in flags and t<r['duration']:
                refs.append((t,v,m,'asr'))
        for k in (4,8):
            for threshold in (.1,.3,.5):
                for gate in ('raw','level'):
                    t=a.get(f'{v}/panns-{k}s/{threshold}/{gate}')
                    if t is not None and t<r['duration']:refs.append((t,v,str(k)+gate,'panns'))
    s=r['smad']; j=s['class_order'].index('singing_music')
    probs=[p[j] for p in s['probabilities']]
    for i in range(1,len(probs)):
        if probs[i]>=.75 and probs[i-1]<.75:
            t=s['window_starts'][i]
            for dt in (0, .5, 1, 1.5):
                if t+dt<r['duration']:refs.append((t+dt,'smad','singing','smad'))
    refs.append((0.,'boundary','zero','boundary'))
    # Retain distinct original onsets; deduplication follows at the cue time.
    return list(dict.fromkeys((round(t,2),v,m,k) for t,v,m,k in refs))

def local(a,t,w):
    if a is None:return [0.,0.,0.]
    curve=a['curves']['vocal']; db=a['rms_db']; ratio=a.get('stem_to_mix_db') or []
    lo=max(0,int((t-w)/a['frame_seconds']));hi=min(len(curve),int((t+w)/a['frame_seconds']))
    if lo>=hi:return [0.,-100.,-100.]
    return [float(max(curve[lo:hi])),float(max(db[lo:hi])),float(max(ratio[lo:hi]) if ratio else -100.)]

def features(r,t):
    a={c['chain']:c['cue_s'] for c in r['candidates']}
    vals=[t,t/30.,float(t<=.2),float(t>=27.)]
    for v in variants:
        for m in ('turbo','qwen'):
            s=r['asr'][v+'/'+m];x=s['first_word_start'];flags=s['quality_flags'] or []
            valid=x is not None and 'invalid_word_timing' not in flags and 'alignment_failed' not in flags and x<r['duration']
            text=s['opening_text'].strip().casefold()
            generic=any(b in text for b in bad)
            vals.extend([float(valid), float(valid and abs(x-t)<=.75),float(valid and abs(x-t)<=2),
                max(-30.,min(30.,t-x)) if valid else 30., float(generic),
                float(len(re.findall(r'\w+',text))>3),float(len(flags)>0)])
        for ctx in (4,):
            x=a.get(f'{v}/panns-{ctx}s/0.1/level')
            y=a.get(f'{v}/panns-{ctx}s/0.1/raw')
            vals.extend([float(x is not None), float(x is not None and abs(x-t)<=.75),float(x is not None and abs(x-t)<=2),
                min(30.,abs(x-t)) if x is not None else 30.,min(30.,abs(y-t)) if y is not None else 30.])
            vals+=local(r['acoustic'].get(f'{v}/{ctx}'),t,.5)
    sm=r['smad'];j=sm['class_order'].index('singing_music')
    active=[p[j] for start,p in zip(sm['window_starts'],sm['probabilities']) if start<=t<=start+4]
    vals += [max(active,default=0.),sum(active)/len(active) if active else 0.]
    av=[];pv=[]
    for v in variants:
        s=r['asr'][v+'/turbo'];x=s['first_word_start']
        if x is not None and x<r['duration']:av.append(x)
        p=a.get(f'{v}/panns-4s/0.1/level')
        if p is not None:pv.append(p)
    vals.extend([min(av,default=30),min(pv,default=30),
                 sum(abs(x-t)<.75 for x in av),sum(abs(x-t)<.75 for x in pv),
                 max((abs(x-y) for x in av for y in pv),default=30.)])
    return vals


def group(row):
    unique = {}
    for cue, *_ in source(row):
        if cue not in unique:
            unique[cue] = features(row, cue)
    if not unique:
        raise ValueError('No candidate times')
    return np.array(list(unique)), np.array(list(unique.values()))


def fit(groups):
    """Learn from labelled recordings; no song identity or labels in the features."""
    x = np.vstack([entry['features'] for entry in groups])
    target = np.concatenate([np.abs(entry['times']-entry['label']) <= 1 for entry in groups]).astype(float)
    model = ExtraTreesRegressor(n_estimators=64, min_samples_leaf=8,
                                max_features=.8, random_state=30, n_jobs=2)
    return model.fit(x, target)


def early_conflict(row, cue):
    """Three distinct stems report an early event far before the cue."""
    candidates = {x['chain']: x['cue_s'] for x in row['candidates']}
    points = [candidates.get(f'{v}/panns-4s/0.1/level') for v in variants]
    points = [p for p in points if p is not None and p < cue - 8]
    return any(sum(abs(x - p) <= .75 for x in points) >= 3 for p in points)
