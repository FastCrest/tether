"""Fetch the official 90-task LIBERO demonstration set and verify its inventory."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path('/data/tether')
os.environ.setdefault('HF_HOME',str(ROOT/'hf'))
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
from huggingface_hub import snapshot_download


def main():
    manifest_path=ROOT/'libero90_download_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    root=Path(manifest['path'])
    if len(manifest['files'])!=90:
        raise RuntimeError('The official manifest must contain all 90 task files.')
    for attempt in range(1,6):
        try:
            snapshot_download(manifest['repo'],repo_type='dataset',revision=manifest['revision'],
                              local_dir=root,allow_patterns='libero_90/*',max_workers=4,token=False)
            break
        except Exception as exc:
            print(f'Download attempt {attempt}: {type(exc).__name__}',flush=True)
            if attempt==5: raise
            time.sleep(3)
    import h5py
    total_demos=0
    verified=[]
    for item in manifest['files']:
        path=root/item['path']
        if not path.is_file() or path.stat().st_size!=item['bytes']:
            raise RuntimeError(f'Missing or wrong-sized task file: {path}')
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
        if item.get('sha256') and digest.hexdigest()!=item['sha256']:
            raise RuntimeError(f'Wrong content hash: {path}')
        with h5py.File(path,'r') as hdf:
            demos=len(hdf['data'])
            if not demos: raise RuntimeError(f'No demonstrations in {path}')
        total_demos+=demos
        verified.append(dict(item,verified_sha256=digest.hexdigest(),demonstrations=demos))
        print(f'Verified {path.name}: {demos} demonstrations',flush=True)
    manifest.update(files=verified,status='complete',verified_tasks=len(verified),
                    total_demonstrations=total_demos,verified_unix_time=time.time())
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(f'Official LIBERO-90 complete: 90 tasks, {total_demos} demonstrations.',flush=True)


if __name__=='__main__': main()
