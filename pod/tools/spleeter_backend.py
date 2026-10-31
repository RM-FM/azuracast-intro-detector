"""Official sherpa-onnx Spleeter 2-stem export, FP16 weights, CPU inference."""
import fcntl,hashlib,json,os,pathlib,shutil,tarfile,time,urllib.request
URL='https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2'
PACK='sherpa-onnx-spleeter-2stems-fp16'
SHA256='d54561979bd2e08a51e7dbd99ac36bb47564e089eefd403636dbca93e811bba2'
FILES=('vocals.fp16.onnx','accompaniment.fp16.onnx')
_sessions={}

def trim_output(vocals,expected):
    if vocals.ndim!=2 or vocals.shape[0]!=2 or not expected<=vocals.shape[1]<=expected+4096:
        raise ValueError(f'Spleeter alignment mismatch: expected (2,{expected}), got {vocals.shape}')
    return vocals[:,:expected]

def ensure_models(cache):
    cache=pathlib.Path(cache);cache.mkdir(parents=True,exist_ok=True)
    with (cache/'.download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=cache/'download.json'
        try:verified=manifest.exists() and json.loads(manifest.read_text()).get('sha256')==SHA256
        except (OSError,ValueError):verified=False
        if verified and all((cache/name).is_file() and (cache/name).stat().st_size>1024 for name in FILES):return cache
        archive=cache/'models.tar.bz2.part'
        print('    Spleeter ONNX: downloading official two-stem model pack (first run only)',flush=True)
        digest=hashlib.sha256()
        try:
            with urllib.request.urlopen(URL,timeout=90) as response,archive.open('wb') as output:
                count=0
                while block:=response.read(1024*1024):
                    output.write(block);digest.update(block);count+=len(block)
                    if count%(5*1024*1024)==0:print(f'    Spleeter download: {count//(1024*1024)} MiB',flush=True)
            if digest.hexdigest()!=SHA256:raise ValueError('Spleeter model pack checksum mismatch; download rejected')
            with tarfile.open(archive,'r:bz2') as bundle:
                for name in FILES:
                    member=bundle.getmember(PACK+'/'+name)
                    if not member.isfile() or not 1024<member.size<100*1024*1024:raise ValueError('Unexpected model archive entry')
                    source=bundle.extractfile(member);temporary=cache/(name+'.part')
                    with source,temporary.open('wb') as target:shutil.copyfileobj(source,target)
                    temporary.replace(cache/name)
            temporary=cache/'download.json.part'
            temporary.write_text(json.dumps({'url':URL,'sha256':digest.hexdigest()},indent=2)+'\n');temporary.replace(manifest)
        finally:archive.unlink(missing_ok=True)
    return cache

def separate(job,decode):
    import numpy as np
    import sherpa_onnx
    import soundfile as sf
    threads=job['threads'];started=time.monotonic()
    if threads not in _sessions:
        directory=ensure_models(pathlib.Path(os.environ.get('POC_MODEL_CACHE','/cache'))/'spleeter-2stems-fp16')
        print(f'    Spleeter ONNX: loading once for this worker; CPU threads={threads}',flush=True)
        config=sherpa_onnx.OfflineSourceSeparationConfig(model=sherpa_onnx.OfflineSourceSeparationModelConfig(
            spleeter=sherpa_onnx.OfflineSourceSeparationSpleeterModelConfig(vocals=str(directory/FILES[0]),accompaniment=str(directory/FILES[1])),
            num_threads=threads,debug=False,provider='cpu'))
        if not config.validate():raise ValueError('Invalid sherpa-onnx Spleeter configuration')
        _sessions[threads]=sherpa_onnx.OfflineSourceSeparation(config)
    load_seconds=time.monotonic()-started
    samples=decode(job['song'],44100,2,job['left'],job['right']-job['left']).reshape(-1,2).T
    if not samples.shape[1]:raise ValueError('Decoded empty Spleeter crop')
    samples=np.ascontiguousarray(samples)
    print(f'    Spleeter ONNX: separating {samples.shape[1]/44100:.2f}s crop',flush=True)
    # The exported model emits whole STFT hops and can truncate a fractional
    # hop at EOF. Supply zero context, then trim, never stretch timestamps.
    padded=np.pad(samples,((0,0),(0,4096)))
    started=time.monotonic();result=_sessions[threads].process(sample_rate=44100,samples=padded)
    inference_seconds=time.monotonic()-started
    if result.sample_rate!=44100 or len(result.stems)!=2:raise ValueError('Unexpected Spleeter output format')
    vocals=np.asarray(result.stems[0].data)
    vocals=trim_output(vocals,samples.shape[1])
    if not np.isfinite(vocals).all():raise ValueError('Non-finite Spleeter output')
    destination=pathlib.Path(job['stem']);temporary=destination.with_suffix('.tmp.wav')
    sf.write(temporary,vocals.T,44100,subtype='FLOAT');temporary.replace(destination)
    print(f'    Spleeter ONNX: load/download {load_seconds:.2f}s, separation {inference_seconds:.2f}s',flush=True)
    return dict(stem=str(destination),runtime='sherpa-onnx '+getattr(sherpa_onnx,'__version__','unknown'),model_sha256=SHA256,separator='spleeter',separation_profile='spleeter2-fp16-v1',load_seconds=round(load_seconds,3),inference_seconds=round(inference_seconds,3),audio_seconds=samples.shape[1]/44100,threads=threads)
