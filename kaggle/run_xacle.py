"""Self-contained Kaggle entrypoint: isolated Python and pinned external sources."""
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import traceback
import tarfile
from pathlib import Path

WORK = Path('/kaggle/working')
OUTPUT = WORK / 'outputs'
ROOT = WORK / 'xacle-src'
PAYLOAD = ''


def unpack():
    ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as archive:
        archive.extractall(ROOT, filter='data')
    sys.path.insert(0, str(ROOT))


def command(*args, cwd=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def cleanup(work):
    """Keep Kaggle output downloads limited to experiment artifacts."""
    for name in ('xacle-env', 'xacle-src', 'm2d', 'MGA-CLAP', 'model_links'):
        path = Path(work) / name
        if path.exists():
            shutil.rmtree(path)


def bootstrap():
    uv = shutil.which('uv')
    if not uv:
        command(sys.executable, '-m', 'pip', 'install', 'uv==0.8.22')
        uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
    command(uv, 'python', 'install', '3.11.13')
    venv = WORK / 'xacle-env'
    command(uv, 'venv', str(venv), '--python', '3.11.13')
    python = venv / 'bin/python'
    command(uv, 'pip', 'install', '--python', str(python), 'torch==2.6.0', 'torchvision==0.21.0', 'torchaudio==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu118')
    command(uv, 'pip', 'install', '--python', str(python), '-r', str(ROOT / 'requirements-kaggle.txt'))
    try:
        command(str(python), str(Path(__file__).resolve()), '--worker')
    finally:
        cleanup(WORK)


def external_sources():
    sources = json.loads((ROOT / 'sources.json').read_text())
    for name, local_name in (('m2d', 'm2d'), ('mga', 'MGA-CLAP')):
        spec = sources[name]
        dest = WORK / local_name
        command('git', 'clone', '--filter=blob:none', spec['url'], str(dest))
        command('git', 'checkout', spec['commit'], cwd=dest)
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dest, text=True).strip()
        if actual != spec['commit']:
            raise RuntimeError(f'{name}: source revision mismatch')
    # MGA inference never calls tools.utils; the wildcard import pulls in
    # optional wandb/sed_scores_eval training dependencies.
    ase = WORK / 'MGA-CLAP/models/ase_model.py'
    contents = ase.read_text()
    line = 'from tools.utils import *'
    if contents.count(line) != 1:
        raise RuntimeError('MGA compatibility patch no longer matches pinned source')
    ase.write_text(contents.replace(line, '# ' + line))
    # PortableM2D infers architecture and CLAP text encoder from parent name.
    # A symlink retains the read-only Kaggle checkpoint without copying it.
    m2d_link = WORK / 'model_links/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025'
    m2d_link.mkdir(parents=True, exist_ok=True)
    return sources, m2d_link


def worker():
    from workflow import validate_config
    cfg = json.loads((ROOT / 'config_kaggle.json').read_text())
    validate_config(cfg)
    mode = json.loads((ROOT / 'mode.json').read_text())['mode']
    if mode == 'train':
        reference = json.loads((ROOT / 'config_submission1.json').read_text())
        original_keys = ('batch_size', 'validation_batch_size', 'num_workers', 'max_len',
                         'sample_rate', 'preprocessor', 'SVR')
        if any(cfg[key] != reference[key] for key in original_keys):
            raise RuntimeError('Kaggle configuration differs from original submission1 training settings')
    sources, link_dir = external_sources()
    m2d_checkpoint = Path(cfg['model']['M2D']['ckpt'])
    (link_dir / m2d_checkpoint.name).symlink_to(m2d_checkpoint)
    cfg['model']['M2D']['ckpt'] = str(link_dir / m2d_checkpoint.name)
    cfg['model']['MGA']['config'] = str(WORK / 'MGA-CLAP/settings/inference_example.yaml')
    sys.path.insert(0, str(WORK))
    sys.path.insert(0, str(WORK / 'm2d'))
    sys.path.insert(0, str(WORK / 'MGA-CLAP'))
    metadata = {'mode': mode, 'sources': sources, 'python': sys.version,
                'checkpoints': {'M2D': str(m2d_checkpoint), 'MGA': cfg['model']['MGA']['ckpt']}}
    (OUTPUT / 'execution_metadata.json').write_text(json.dumps(metadata, indent=2))
    import importlib.metadata
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'torchaudio', 'numpy', 'transformers', 'msclap', 'timm', 'scikit-learn')}
    (OUTPUT / 'dependency_versions.json').write_text(json.dumps(packages, indent=2))
    if mode == 'preflight':
        import torch
        from model_preflight import run
        run(cfg, OUTPUT)
    elif mode == 'train':
        os.environ['XACLE_LOG_PATH'] = str(OUTPUT / 'execution.log')
        from dual_gpu_pipeline import run
        run(cfg, OUTPUT, sources, ROOT)
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    OUTPUT.mkdir(exist_ok=True)
    try:
        unpack()
        if '--worker' in sys.argv:
            worker()
        else:
            bootstrap()
    except Exception as exc:
        (OUTPUT / 'failure.json').write_text(json.dumps({'error': repr(exc), 'traceback': traceback.format_exc()}, indent=2))
        raise
