"""Remote data and feature extractor checks before training."""
import gc
import hashlib
import importlib
import json
import os
import traceback
import stat
from pathlib import Path


def check_dataset(cfg):
    import pandas as pd
    import soundfile as sf
    report = {}
    seen = {}
    for split in ('train', 'validation', 'test', 'test_average'):
        csv_path = Path(cfg[f'{split}_list'])
        frame = pd.read_csv(csv_path)
        required = {'wav_file_name', 'text'} | ({'average_score'} if split != 'test' else set())
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{csv_path}: missing columns {sorted(missing)}; columns={list(frame.columns)}')
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError(f'{csv_path}: empty rows or missing required values')
        if split != 'test' and not frame['average_score'].between(0, 10).all():
            raise ValueError(f'{csv_path}: average_score outside [0, 10]')
        wav_dir = Path(cfg['wav_dir']) / ('test' if split == 'test_average' else split)
        names = set()
        wav_identity = hashlib.sha256()
        for name in frame['wav_file_name']:
            name = str(name)
            path = wav_dir / name.lstrip('/')
            try:
                file_stat = path.stat()
            except FileNotFoundError:
                raise FileNotFoundError(f'{csv_path}: missing audio {path}')
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f'{csv_path}: audio path is not a file: {path}')
            wav_identity.update(f'{name}:{file_stat.st_size}:{file_stat.st_mtime_ns}\n'.encode())
            names.add(path.name)
        audio = wav_dir / str(frame.iloc[0]['wav_file_name']).lstrip('/')
        info = sf.info(audio)
        report[split] = {'rows': len(frame), 'columns': list(frame.columns), 'sample_audio': str(audio),
                         'sample_rate': info.samplerate, 'channels': info.channels,
                         'wav_manifest_sha256': wav_identity.hexdigest()}
        seen[split] = names
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        overlap = seen[a] & seen[b]
        if overlap:
            raise ValueError(f'{a}/{b} split overlap: {len(overlap)} audio names; example {next(iter(overlap))}')
    if seen['test_average'] != seen['test'] or report['test_average']['rows'] != len(seen['test_average']):
        raise ValueError('test_average.csv must contain one row for each test WAV')
    return report


def run(cfg, output_dir):
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets.fetch_data import get_dataset
    from features.all_feature_dict import GetFeatures
    report = {'dataset': None, 'packages': {}, 'models': {}, 'cuda': {}, 'checkpoints': {}}
    output_dir = Path(output_dir)
    def save():
        (output_dir / 'preflight_report.json').write_text(json.dumps(report, indent=2, default=str))
    try:
        report['dataset'] = check_dataset(cfg)
        for key in ('M2D', 'MGA'):
            path = Path(cfg['model'][key]['ckpt'])
            if not path.is_file():
                raise FileNotFoundError(path)
            report['checkpoints'][key] = {'path': str(path), 'bytes': path.stat().st_size}
        for package in ('torch', 'torchvision', 'torchaudio', 'numpy', 'transformers', 'msclap', 'timm', 'nnAudio', 'sentence_transformers', 'sklearn'):
            module = importlib.import_module(package)
            report['packages'][package] = getattr(module, '__version__', 'unknown')
        report['cuda'] = {'available': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU fallback is prohibited')
        dataset = get_dataset(cfg['train_list'], os.path.join(cfg['wav_dir'], 'train'), cfg['max_len'], cfg['sample_rate'])
        loader = DataLoader(Subset(dataset, [0]), batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)
        for model in ('M2D_Clap', 'MS_Clap', 'Laion_Clap', 'MGA_Clap', 'Whisper'):
            try:
                features, labels, parameters = GetFeatures(model, loader, 'preflight', cfg, False)
                report['models'][model] = {'status': 'ok', 'parameters': parameters, 'shape': list(features['AudioFeatures'].shape), 'gpu_peak_bytes': torch.cuda.max_memory_allocated()}
            except Exception as exc:
                report['models'][model] = {'status': 'error', 'error': repr(exc), 'traceback': traceback.format_exc()}
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                save()
        if any(item['status'] != 'ok' for item in report['models'].values()):
            raise RuntimeError('One or more model preflight checks failed; see preflight_report.json')
    except Exception as exc:
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        save()
        raise
    save()
    return report
