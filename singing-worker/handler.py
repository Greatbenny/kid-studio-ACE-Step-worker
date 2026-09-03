import base64, hashlib, json, os, secrets, shutil, subprocess, tempfile
from pathlib import Path
from typing import Any
import runpod, torch

SERVICE="kid-studio-seed-vc-worker"
BUILD="seed-vc-singing-v1"
SEED_COMMIT="51383efd921027683c89e5348211d93ff12ac2a8"
DEMUCS_COMMIT="2883f3db65617d6d178c6ed10d869dc14e44e59b"
SEED_ROOT=Path(os.getenv("SEED_VC_ROOT","/opt/seed-vc"))
TMP_ROOT=Path(os.getenv("TMPDIR","/runpod-volume/tmp"))
MAX_AUDIO_BYTES=int(os.getenv("MAX_AUDIO_BYTES",str(20*1024*1024)))
MAX_DURATION=float(os.getenv("MAX_DURATION_SECONDS","240"))

def _decode(value:Any,name:str)->bytes:
    if not isinstance(value,str) or not value.strip(): raise ValueError(f"{name} is required.")
    try: raw=base64.b64decode(value,validate=True)
    except Exception as exc: raise ValueError(f"{name} is not valid base64.") from exc
    if not raw or len(raw)>MAX_AUDIO_BYTES: raise ValueError(f"{name} is empty or too large.")
    return raw

def _run(args:list[str],cwd:Path|None=None,timeout:int=1200)->None:
    result=subprocess.run(args,cwd=cwd,text=True,capture_output=True,timeout=timeout)
    if result.returncode: raise RuntimeError((result.stderr or result.stdout)[-4000:])

def _duration(path:Path)->float:
    result=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(path)],text=True,capture_output=True,check=True)
    return float(result.stdout.strip())

def _convert(source:Path,target:Path,outdir:Path,semitones:int,steps:int)->Path:
    before=set(outdir.glob("*.wav"))
    _run(["python","inference.py","--source",str(source),"--target",str(target),"--output",str(outdir),"--diffusion-steps",str(steps),"--length-adjust","1.0","--inference-cfg-rate","0.7","--f0-condition","True","--auto-f0-adjust","False","--semi-tone-shift",str(semitones),"--fp16","True"],SEED_ROOT)
    created=sorted(set(outdir.glob("*.wav"))-before,key=lambda p:p.stat().st_mtime)
    if not created: raise RuntimeError("Seed-VC produced no converted vocal.")
    return created[-1]

def _health()->dict[str,Any]:
    props=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    return {"ok":True,"service":SERVICE,"worker_build":BUILD,"seed_vc_commit":SEED_COMMIT,"demucs_commit":DEMUCS_COMMIT,"gpu":{"available":bool(props),"name":props.name if props else None,"vram_bytes":props.total_memory if props else None}}

def _generate(data:dict[str,Any])->dict[str,Any]:
    if data.get("voice_clone_consent") is not True: raise ValueError("Explicit voice-cloning consent is required.")
    performers=data.get("performers")
    if not isinstance(performers,list) or not performers: raise ValueError("performers must be a non-empty list.")
    TMP_ROOT.mkdir(parents=True,exist_ok=True)
    job=Path(tempfile.mkdtemp(prefix="seed-vc-",dir=TMP_ROOT))
    try:
        guide=job/"guide.mp3"; guide.write_bytes(_decode(data.get("guide_audio"),"guide_audio"))
        duration=_duration(guide)
        if duration<=0 or duration>MAX_DURATION: raise ValueError("guide_audio duration is outside the configured limit.")
        separated=job/"separated"
        _run(["python","-m","demucs.separate","--two-stems=vocals","-n","htdemucs","-o",str(separated),str(guide)])
        stemdir=separated/"htdemucs"/guide.stem
        vocals=stemdir/"vocals.wav"; instrumental=stemdir/"no_vocals.wav"
        if not vocals.is_file() or not instrumental.is_file(): raise RuntimeError("Stem separation produced incomplete output.")
        mix_inputs=[instrumental]; filters=[]; labels=[]
        steps=max(4,min(50,int(data.get("diffusion_steps") or 35)))
        for pi,performer in enumerate(performers):
            if not isinstance(performer,dict): raise ValueError("Each performer must be an object.")
            identity=str(performer.get("character_asset_name") or '').strip()
            if not identity: raise ValueError("Every performer requires character_asset_name.")
            reference=job/f"reference-{pi}.wav"; reference.write_bytes(_decode(performer.get("reference_audio"),f"reference_audio for {identity}"))
            segments=performer.get("segments")
            if not isinstance(segments,list) or not segments: raise ValueError(f"{identity} requires timed segments.")
            for si,segment in enumerate(segments):
                start=float(segment.get("start_seconds",-1)); end=float(segment.get("end_seconds",-1))
                if start<0 or end<=start or end>duration+0.25: raise ValueError(f"Invalid segment for {identity}.")
                source=job/f"source-{pi}-{si}.wav"
                _run(["ffmpeg","-y","-ss",str(start),"-to",str(end),"-i",str(vocals),"-ar","44100","-ac","1",str(source)],timeout=120)
                converted=_convert(source,reference,job,int(segment.get("semitones") or 0),steps)
                mix_inputs.append(converted); idx=len(mix_inputs)-1
                gain=max(-24.0,min(12.0,float(segment.get("gain_db") or 0)))
                pan=max(-1.0,min(1.0,float(segment.get("pan") or 0)))
                left=(1-pan)/2; right=(1+pan)/2; delay=round(start*1000)
                label=f"v{pi}_{si}"; labels.append(f"[{label}]")
                filters.append(f"[{idx}:a]adelay={delay}|{delay},volume={gain}dB,pan=stereo|c0={left}*c0|c1={right}*c0[{label}]")
        output=job/"final.mp3"
        inputs=sum((["-i",str(p)] for p in mix_inputs),[])
        chain=";".join(filters+[f"[0:a]{''.join(labels)}amix=inputs={1+len(labels)}:duration=first:normalize=0,alimiter=limit=0.95[out]"])
        _run(["ffmpeg","-y",*inputs,"-filter_complex",chain,"-map","[out]","-c:a","libmp3lame","-b:a","192k",str(output)])
        raw=output.read_bytes()
        return {**_health(),"audio_base64":base64.b64encode(raw).decode(),"mime_type":"audio/mpeg","duration_seconds":duration,"audio_sha256":hashlib.sha256(raw).hexdigest(),"performer_count":len(performers)}
    finally: shutil.rmtree(job,ignore_errors=True)

def handler(job:dict[str,Any])->dict[str,Any]:
    data=job.get("input")
    if not isinstance(data,dict): return {"ok":False,"error":"input must be an object."}
    try:
        operation=str(data.get("operation") or "generate").lower()
        if operation in {"health","preflight"}: return _health()
        if operation!="generate": raise ValueError("operation must be health, preflight, or generate.")
        return _generate(data)
    except Exception as exc: return {"ok":False,"service":SERVICE,"worker_build":BUILD,"error":str(exc),"error_type":type(exc).__name__}

if __name__=="__main__": runpod.serverless.start({"handler":handler})
