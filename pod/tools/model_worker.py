#!/usr/bin/env python3
"""Sequential ML stages; optional request server retains models across waves."""
import argparse,contextlib,json,os,pathlib,resource,subprocess,sys,time

def decode(path,rate,channels=1,start=None,duration=None):
    import numpy as np
    cmd=['ffmpeg','-v','error','-threads','1']
    if start is not None:cmd+=['-ss',str(start)]
    cmd+=['-i',str(path)]
    if duration is not None:cmd+=['-t',str(duration)]
    raw=subprocess.check_output(cmd+['-f','f32le','-ac',str(channels),'-ar',str(rate),'-'])
    return np.frombuffer(raw,dtype='<f4').copy()

_panns=None
_panns_device=None

def separate(job):
    if job.get('separator','spleeter')!='spleeter':raise ValueError('Only Spleeter separation is supported')
    from spleeter_backend import separate as run_spleeter
    return run_spleeter(job,decode)

def classify(job):
    global _panns,_panns_device
    import numpy as np
    import torch
    from panns_inference import SoundEventDetection,labels
    from frame_alignment import align_frames
    from vocal_fusion import STEM_EDGE_PROFILE,classifier_regions
    device=job.get('classifier_device','cpu')
    if device not in ('cpu','cuda'):raise ValueError('Invalid classifier device')
    if _panns is not None and _panns_device!=device:
        raise ValueError('Cannot change classifier device within a worker')
    torch.set_num_threads(job['threads'])
    wanted={'Speech','Singing','Male singing','Female singing','Child singing','Choir','Rapping','Humming','Chant','Mantra','Vocal music'}
    indices=[i for i,x in enumerate(labels) if x in wanted]
    if not indices:raise RuntimeError('No PANNs vocal labels found')
    mono=decode(job['stem'],32000)
    if not len(mono):raise ValueError('Empty vocal stem')
    frames=int(np.ceil(len(mono)/320));scores=np.zeros(frames,dtype=np.float32)
    blocks=np.pad(mono,(0,frames*320-len(mono))).reshape(-1,320)
    db=20*np.log10(np.maximum(np.sqrt((blocks**2).mean(axis=1)),1e-10))
    input_seconds=job.get('classifier_input_seconds',4)
    if input_seconds not in (4,8):raise ValueError('Classifier input must be 4 or 8 seconds')
    input_samples=input_seconds*32000;core_samples=input_samples//2;context_samples=input_samples//4
    required=[(0,frames)] if job.get('full_classification',False) else classifier_regions(db.tolist(),frames)
    coverage=[];windows=0
    # Bound CNN feature-map memory: 2s of judged audio + up to 1s context
    # per side, padded to 4s. Never submit an entire 20s crop to PANNs.
    for core in range(0,len(mono),core_samples):
        end=min(len(mono),core+core_samples);left=max(0,core-context_samples);right=min(len(mono),end+context_samples)
        length=int(np.ceil((end-core)/320));offset=core//320
        if not any(a<offset+length and b>offset for a,b in required):continue
        if _panns is None:
            print('    PANNs: loading weights once for this worker',flush=True)
            from checkpoint_loading import load_weights_only
            _panns=load_weights_only(SoundEventDetection,checkpoint_path=None,device=device)
            _panns_device=device
        clip=mono[left:right];clip=np.pad(clip,(0,input_samples-len(clip)))
        with torch.inference_mode():prediction=_panns.inference(clip[None])[0][:,indices].max(axis=1)
        aligned,_=align_frames(prediction.tolist(),[-100.]*(input_seconds*100),input_seconds*100)
        lo=(core-left)//320
        scores[offset:offset+length]=aligned[lo:lo+length]
        if coverage and coverage[-1][1]==offset:coverage[-1][1]=offset+length
        else:coverage.append([offset,offset+length])
        windows+=1
    total=(len(mono)+core_samples-1)//core_samples
    print(f'    PANNs: classified {windows}/{total} bounded {input_seconds}s inputs; unused cores skipped',flush=True)
    return {'classifier_input_seconds':input_seconds,'stem_edge_profile':STEM_EDGE_PROFILE,'scores':scores.tolist(),'rms_db':db.tolist(),
        'labels':[labels[i] for i in indices],'classified_ranges':coverage,'classifier_windows':windows,'classifier_total_windows':total}

def execute(stage,job):
    if stage not in ('separate','classify'):raise ValueError('Unsupported model stage')
    started=time.monotonic()
    if 'items' in job:
        reports=[]
        for i,item in enumerate(job['items'],1):
            item_started=time.monotonic()
            print(f'    {stage}: batch item {i}/{len(job["items"])}',flush=True)
            try:
                value=globals()[stage](item)
                value['elapsed_seconds']=round(time.monotonic()-item_started,2)
                print(f'    {stage}: item complete in {value["elapsed_seconds"]}s',flush=True)
                value['worker_peak_mib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)
                target=pathlib.Path(item['result_path']);temp=target.with_suffix('.tmp')
                temp.write_text(json.dumps(value));temp.replace(target)
                reports.append({'result_path':str(target),'ok':True})
            except Exception as error:
                reports.append({'result_path':item['result_path'],'ok':False,'error':str(error)})
                print(f'    {stage}: ERROR {error}',flush=True)
        output={'items':reports}
    else:output=globals()[stage](job)
    output['worker_peak_mib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1)
    output['elapsed_seconds']=round(time.monotonic()-started,2)
    print(f'    {stage}: complete, process peak {output["worker_peak_mib"]} MiB',flush=True)
    return output

def serve():
    # stdout is a JSON-only response channel; model/library logs go to stderr.
    for line in sys.stdin:
        try:
            request=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                output=execute(request['stage'],request['job'])
            response={'ok':True,'output':output}
        except Exception as error:
            response={'ok':False,'error':str(error)}
        print(json.dumps(response),flush=True)

def main():
    if sys.argv[1:]==['--serve']:
        serve();return
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['separate','classify']);ap.add_argument('job');ap.add_argument('output');args=ap.parse_args()
    output=execute(args.stage,json.loads(pathlib.Path(args.job).read_text()))
    target=pathlib.Path(args.output);temp=target.with_suffix('.tmp');temp.write_text(json.dumps(output));temp.replace(target)
if __name__=='__main__':main()

