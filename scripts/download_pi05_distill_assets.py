"""Download pinned SnapFlow inputs to /data without downloading a student."""
import json
import argparse
import hashlib
import os
from pathlib import Path

ROOT = Path('/data/tether')
os.environ.setdefault('HF_HOME', str(ROOT / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import HfApi, snapshot_download

SOURCES = [
    ('lerobot/pi05_libero_finetuned_v044', 'model',
     '7152f13e34c6c115d808aea0e5d08aca74eac14b',
     ROOT / 'checkpoints/pi05_libero_finetuned_v044', None),
    ('lerobot/libero', 'dataset',
     'a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4',
     ROOT / 'lerobot/lerobot/libero', None),
    # Public tokenizer copy: all five Git blobs match Google's pinned original.
    ('leo009/paligemma-3b-pt-224', 'model',
     '39996beb6fb17c5d16a50d3ef8f7a96ad9d03986',
     ROOT / 'tokenizers/paligemma-3b-pt-224',
     ['added_tokens.json', 'special_tokens_map.json', 'tokenizer.json',
      'tokenizer.model', 'tokenizer_config.json']),
]
TOKENIZER_HASHES = {
    'added_tokens.json': ('sha1', 'ff9246e420971b10c510b39a0de289132c2bf23f'),
    'special_tokens_map.json': ('sha1', '0c18fdd94629ea652e2ef36ed9a422474ad0a8e8'),
    'tokenizer_config.json': ('sha1', '960dcec0bccc6e594d865eceb63c8078e164cfb2'),
    'tokenizer.json': ('sha256', 'ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563'),
    'tokenizer.model': ('sha256', '8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-only', action='store_true',
                        help='Download teacher and dataset; omit tokenizer files.')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest = []
    for repo, kind, revision, dest, selected in SOURCES:
        if args.inputs_only and selected is not None:
            continue
        info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
        files = [s for s in info.siblings if selected is None or s.rfilename in selected]
        print(f'Downloading {repo}@{revision} -> {dest}: '
              f'{sum(s.size or 0 for s in files) / 2**30:.3f} GiB', flush=True)
        snapshot_download(repo, repo_type=kind, revision=revision,
                          local_dir=dest, allow_patterns=selected, max_workers=4)
        for f in files:
            p = dest / f.rfilename
            if not p.is_file() or (f.size is not None and p.stat().st_size != f.size):
                raise RuntimeError(f'Missing or wrong-sized file: {p}')
        if selected:
            for name, (algorithm, expected) in TOKENIZER_HASHES.items():
                payload = (dest / name).read_bytes()
                if algorithm == 'sha1':
                    payload = b'blob ' + str(len(payload)).encode() + b'\0' + payload
                if hashlib.new(algorithm, payload).hexdigest() != expected:
                    raise RuntimeError(f'Tokenizer differs from original Google file: {name}')
        manifest.append(dict(repo=repo, repo_type=kind, revision=revision,
                             path=str(dest), bytes=sum(s.size or 0 for s in files),
                             files=[dict(path=s.rfilename, bytes=s.size) for s in files]))
        (ROOT / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'Verified {len(files)} files for {repo}', flush=True)
    print('All inputs downloaded and file sizes verified.', flush=True)


if __name__ == '__main__':
    main()
