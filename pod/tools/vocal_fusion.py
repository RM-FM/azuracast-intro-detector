"""Acoustic regions corroborated by vocal classification; no ML dependencies."""
import math, statistics

FUSION_POLICY='intro-uncertainty-context-v3-unfaded'
STEM_EDGE_PROFILE='stem-unfaded-v1'

def percentile(values,fraction):
    values=sorted(values)
    return values[min(len(values)-1,int((len(values)-1)*fraction))]

def ranges(mask,step=.1,gap=.4,minimum=.3):
    out=[]; start=None
    for i,active in enumerate(list(mask)+[False]):
        if active and start is None:start=i
        if not active and start is not None:
            out.append([start*step,i*step]);start=None
    merged=[]
    for a,b in out:
        if merged and a-merged[-1][1]<=gap+1e-6:merged[-1][1]=b
        else:merged.append([a,b])
    return [[round(a,3),round(b,3)] for a,b in merged if b-a>=minimum-1e-6]

def activity(db):
    if not db:return {'balanced':[],'sensitive':[],'strict':[]}
    # 0.4-second centered RMS, sampled every 0.1s; keep the same time origin.
    power=[10**(x/10) for x in db]; prefix=[0.]
    for x in power:prefix.append(prefix[-1]+x)
    env=[]
    for i in range(0,len(db),10):
        a=max(0,i-20);b=min(len(db),i+20)
        env.append(10*math.log10(max((prefix[b]-prefix[a])/(b-a),1e-20)))
    env=[statistics.median(env[max(0,i-1):i+2]) for i in range(len(env))]
    peak=max(env)
    if peak<-70:return {'balanced':[],'sensitive':[],'strict':[]}
    relative=[x-peak for x in env]; floor=percentile(relative,.15)
    # A continuous vocal chunk has no noise-floor sample. Do not place the
    # activity threshold above the chunk peak in that case.
    thresholds={k:min(-6,max(base,floor+delta)) for k,base,delta in [('strict',-32,14),('balanced',-40,10),('sensitive',-48,7)]}
    result={k:ranges([x>=v for x in relative]) for k,v in thresholds.items()}
    result['thresholds']=thresholds
    return result

def classifier_regions(db,frames):
    """Exact half-open score ranges consumed by fusion, in 10 ms frames."""
    return [(max(0,int(a*100)),min(frames,math.ceil(b*100))) for a,b in activity(db)['balanced']]

def classified_ranges(data):
    frames=len(data['scores'])
    coverage=data.get('classified_ranges',[[0,frames]] if frames else [])
    previous=0
    for a,b in coverage:
        if type(a) is not int or type(b) is not int or not previous<=a<b<=frames:
            raise ValueError('Invalid classifier frame coverage')
        previous=b
    return coverage

def classification_covers_activity(data,db):
    coverage=classified_ranges(data)
    for a,b in classifier_regions(db,len(data['scores'])):
        cursor=a
        for left,right in coverage:
            if left>cursor:break
            if right>cursor:cursor=right
            if cursor>=b:break
        if cursor<b:return False
    return True

def classification_is_current(data,db):
    # Unlabelled caches predate the rejected fade experiment and used raw stems.
    return data.get('stem_edge_profile',STEM_EDGE_PROFILE)==STEM_EDGE_PROFILE and classification_covers_activity(data,db)

def fuse_without_asr(db,scores):
    """Classifier corroboration + adaptive acoustic boundaries, no transcription.

    Scores are model outputs, not calibrated probabilities of overlay safety.
    """
    act=activity(db);events=[];intro_events=[]
    for a,b in act['balanced']:
        evidence=scores[max(0,int(a*100)):min(len(scores),math.ceil(b*100))]
        support=ranges([x>=.10 for x in evidence],step=.01,gap=.1,minimum=.3)
        weak=any(x>=.10 for x in evidence)
        if support:
            tail=b
            for c,d in act['sensitive']:
                if c<=b and d>=a:tail=max(tail,d)
            events.append(dict(start=a,end=tail,accepted=True,
                method='classifier+adaptive-activity',anchors=[],
                classifier_peak=max(evidence),classifier_threshold=.10,
                support_ranges=[[round(a+c,3),round(a+d,3)] for c,d in support]))
        elif weak:
            events.append(dict(start=a,end=b,accepted=False,
                method='brief-unconfirmed-classifier-response',anchors=[],
                classifier_peak=max(evidence),classifier_threshold=.10))
        else:
            # Weaker scores need longer support than accepted vocals so a
            # brief crop-edge artifact cannot veto a previously clean intro.
            uncertain=ranges([x>=.05 for x in evidence],step=.01,gap=.1,minimum=1.)
            if uncertain:
                intro_events.append(dict(start=a,end=b,accepted=False,
                    method='sustained-subthreshold-classifier-response',anchors=[],
                    classifier_peak=max(evidence),classifier_threshold=.10,
                    uncertainty_threshold=.05,
                    support_ranges=[[round(a+c,3),round(a+d,3)] for c,d in uncertain]))
    # Subthreshold evidence can veto an intro but never grants overlay safety.
    intro_events.extend(dict(event) for event in events)
    intro_events.sort(key=lambda event:event['start'])
    return {'events':events,'intro_events':intro_events,'activity':act,'fusion_policy':FUSION_POLICY}

def confirm_intro(evidence,db,scores):
    """Corroborate the same first region, never skip it for a later phrase."""
    if not evidence['intro_events']:return
    first=evidence['intro_events'][0]
    if first['accepted']:return
    confirmation=fuse_without_asr(db,scores)
    for event in confirmation['events']:
        if event['accepted'] and abs(event['start']-first['start'])<1e-6:
            evidence['intro_events'][0]={**event,
                'method':'classifier+adaptive-activity+8s-confirmation',
                'primary_classifier_peak':first['classifier_peak'],
                'confirmation_input_seconds':8}
            break
