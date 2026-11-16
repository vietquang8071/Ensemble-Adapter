"""Local validation, source staging, and Kaggle CLI control."""
import argparse
import base64
import io
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL = 'qnv8071/xacle-ensemble-training'
SOURCE_FILES = ('train.py', 'inference.py', 'evaluate.py', 'config_kaggle.json', 'config_submission1.json')
SOURCE_DIRS = ('datasets', 'features', 'load_pretrained_models', 'utils')


def validate_config(cfg):
    expected_data = '/kaggle/input/datasets/qnv8071/xacle-2027-data/XACLE_Challenge_2027_development_dataset'
    expected_weights = '/kaggle/input/datasets/qnv8071/xacle-weights/weights/pretrained'
    for split, filename in (('train', 'train_average.csv'), ('validation', 'validation_average.csv'), ('test', 'test.csv')):
        if cfg.get(split + '_list') != f'{expected_data}/meta_data/{filename}':
            raise ValueError(f'{split}_list does not match the verified Kaggle mount')
    if cfg.get('test_average_list') != f'{expected_data}/meta_data/test_average.csv':
        raise ValueError('test_average_list does not match the verified Kaggle mount')
    if cfg.get('wav_dir') != f'{expected_data}/wav':
        raise ValueError('wav_dir does not match the verified Kaggle mount')
    if cfg.get('output_dir') != '/kaggle/working/outputs':
        raise ValueError('output_dir must be /kaggle/working/outputs')
    if cfg.get('device') != 'cuda':
        raise ValueError('device must be cuda')
    for key, filename in (('M2D', 'checkpoint-30.pth'), ('MGA', 'mga-clap.pt')):
        if cfg['model'][key]['ckpt'] != f'{expected_weights}/{filename}':
            raise ValueError(f'{key} checkpoint path is invalid')


def stage_kernel(root, destination, mode):
    if mode not in ('preflight', 'train'):
        raise ValueError('mode must be preflight or train')
    destination.mkdir(parents=True, exist_ok=False)
    for name in SOURCE_FILES:
        shutil.copy2(root / name, destination / name)
    for name in SOURCE_DIRS:
        shutil.copytree(root / name, destination / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (destination / name / '__init__.py').touch(exist_ok=True)
    for name in ('kernel-metadata.json', 'run_xacle.py', 'model_preflight.py', 'dual_gpu.py',
                 'dual_gpu_pipeline.py', 'requirements-kaggle.txt', 'sources.json', 'workflow.py'):
        shutil.copy2(root / 'kaggle' / name, destination / name)
    (destination / 'mode.json').write_text(json.dumps({'mode': mode}))
    # Kaggle CLI's script push sends only code_file, not sibling files.
    # Embed the exact staged sources in that single Python file.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in sorted(destination.rglob('*')):
            if path.is_file() and path.name not in ('run_xacle.py', 'kernel-metadata.json'):
                archive.add(path, arcname=path.relative_to(destination))
    script = (destination / 'run_xacle.py').read_text()
    marker = "PAYLOAD = ''"
    if script.count(marker) != 1:
        raise RuntimeError('entrypoint payload marker changed')
    (destination / 'run_xacle.py').write_text(script.replace(marker, 'PAYLOAD = ' + repr(base64.b64encode(buffer.getvalue()).decode())))
    return destination


def run(args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(args, check=True)


def push_command(stage, publish=False):
    command = ['kaggle', 'kernels', 'push', '-p', str(stage), '--accelerator', 'NvidiaTeslaT4']
    return command + (['--no-run'] if publish else [])


def verify_outputs(target):
    target = Path(target)
    metrics = list(target.rglob('metrics.json'))
    if metrics:
        root = metrics[0].parent
        required = ['config.json', 'train_predictions.csv', 'validation_predictions.csv',
                    'test_predictions.csv', 'test_metrics.json', 'inference_result_for_test.csv']
        for ensemble in ('A', 'B'):
            required.extend([f'{ensemble}/SVR_{ensemble}.pkl', f'{ensemble}/Normalizer_{ensemble}.pkl', f'{ensemble}/StandardScaler_{ensemble}.pkl'])
        for name in required:
            if not (root / name).is_file():
                raise ValueError(f'Download incomplete: missing {name}')
        scores = json.loads(metrics[0].read_text())
        if not all(key in scores for key in ('train', 'validation', 'weights')):
            raise ValueError('Training metrics incomplete')
        report_path = root.parent / 'execution_report.json'
        if not report_path.is_file():
            raise ValueError('Execution report missing')
        report = json.loads(report_path.read_text())
        if report.get('test_inference') != 'complete' or 'test' not in report.get('metrics', {}):
            raise ValueError('Test evaluation incomplete')
        return 'training'
    reports = list(target.rglob('preflight_report.json'))
    if reports:
        report = json.loads(reports[0].read_text())
        if len(report.get('models', {})) != 5 or any(v.get('status') != 'ok' for v in report['models'].values()):
            raise ValueError('Preflight did not pass all five models')
        return 'preflight'
    raise ValueError('No preflight report or training metrics in downloaded outputs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('validate', 'publish', 'preflight', 'train', 'status', 'logs', 'download'))
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config_kaggle.json').read_text())
    validate_config(cfg)
    if args.command == 'validate':
        metadata = json.loads((ROOT / 'kaggle' / 'kernel-metadata.json').read_text())
        assert metadata['id'] == KERNEL and metadata['is_private'] and metadata['enable_gpu']
        print('Local configuration and kernel metadata valid. Remote CSV and checkpoints require preflight.')
    elif args.command in ('publish', 'preflight', 'train'):
        # The CLI only submits once per explicit command. No automatic retry.
        if args.command != 'publish':
            status = subprocess.run(['kaggle', 'kernels', 'status', KERNEL], text=True, capture_output=True)
            if status.returncode == 0 and any(word in status.stdout.upper() for word in ('RUNNING', 'QUEUED', 'PENDING')):
                raise RuntimeError('The kernel already has an active run; refusing another GPU submission')
        stage = ROOT / '.kaggle-stage' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        stage_kernel(ROOT, stage, 'preflight' if args.command == 'publish' else args.command)
        print(f'Staged {args.command} at {stage}')
        command = push_command(stage, args.command == 'publish')
        print('+', ' '.join(command), flush=True)
        result = subprocess.run(command, text=True, capture_output=True)
        print(result.stdout, end='')
        print(result.stderr, end='')
        if result.returncode or 'Kernel push error:' in result.stdout:
            raise RuntimeError('Kaggle kernel push failed; see output above')
    elif args.command == 'status':
        run(['kaggle', 'kernels', 'status', KERNEL])
    elif args.command == 'logs':
        run(['kaggle', 'kernels', 'logs', KERNEL] + (['--follow'] if args.follow else []))
    elif args.command == 'download':
        target = ROOT / 'results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target.mkdir(parents=True, exist_ok=False)
        run(['kaggle', 'kernels', 'output', KERNEL, '-p', str(target), '--file-pattern', '^outputs/', '--page-size', '200'])
        print(f'Downloaded to {target}', flush=True)
        kind = verify_outputs(target)
        print(f'Verified {kind} artifacts in {target}')


if __name__ == '__main__':
    main()
