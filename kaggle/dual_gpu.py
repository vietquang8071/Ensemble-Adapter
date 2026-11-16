"""Two-process CUDA extraction with ordered, validated feature caches."""
import gc
import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODELS = ('M2D_Clap', 'MS_Clap', 'Laion_Clap', 'MGA_Clap', 'Whisper')
SLUGS = {'M2D_Clap': 'm2d_clap', 'MS_Clap': 'ms_clap', 'Laion_Clap': 'laion_clap',
         'MGA_Clap': 'mga_clap', 'Whisper': 'whisper'}
DIMS = {'M2D_Clap': 768, 'MS_Clap': 1024, 'Laion_Clap': 768, 'MGA_Clap': 1024, 'Whisper': 1280}
BATCH_SIZES = {name: (1 if name == 'Whisper' else 8) for name in MODELS}
ARRAY_KEYS = ('indices', 'wav_file_name', 'labels', 'AudioFeatures', 'TextFeatures',
              'Cosine_Sim', 'Cosine_Ang', 'L1', 'L2')


def log(message):
    line = f'[{datetime.now(timezone.utc).isoformat()}] {message}'
    print(line, flush=True)
    if os.environ.get('XACLE_LOG_PATH'):
        with open(os.environ['XACLE_LOG_PATH'], 'a', encoding='utf-8') as file:
            file.write(line + '\n')


def partition_indices(count, batch_size):
    if count < 2:
        raise ValueError('Need at least two samples to use both GPUs')
    boundary = round((count / 2) / batch_size) * batch_size
    boundary = min(max(boundary, 1), count - 1)
    return list(range(boundary)), list(range(boundary, count))


def add_proximity(audio, text):
    from features import proximity_features as proximity
    return {'Cosine_Sim': proximity.cosine_similarity(audio, text),
            'Cosine_Ang': proximity.angular_distance(audio, text),
            'L1': proximity.L1_normalized(audio, text),
            'L2': proximity.L2_normalized(audio, text)}


def merge_shards(shards, expected_names, expected_labels, feature_dim):
    if len(shards) != 2:
        raise ValueError('Exactly two GPU shards are required')
    keys = ('indices', 'wav_file_name', 'labels', 'AudioFeatures', 'TextFeatures')
    if any(any(key not in shard for key in keys) for shard in shards):
        raise ValueError('Shard is missing required arrays')
    combined = {key: np.concatenate([shard[key] for shard in shards]) for key in keys}
    order = np.argsort(combined['indices'])
    combined = {key: value[order] for key, value in combined.items()}
    n = len(expected_names)
    if len(combined['indices']) != n or not np.array_equal(combined['indices'], np.arange(n)):
        raise ValueError('Shards contain duplicate or missing sample indices')
    if not np.array_equal(combined['wav_file_name'].astype(str), np.asarray(expected_names).astype(str)):
        raise ValueError('Merged sample names do not match original order')
    if not np.allclose(combined['labels'], expected_labels, atol=1e-6, rtol=0):
        raise ValueError('Merged labels do not align with metadata')
    for key in ('AudioFeatures', 'TextFeatures'):
        value = combined[key]
        if value.shape != (n, feature_dim) or value.dtype.kind != 'f' or not np.isfinite(value).all():
            raise ValueError(f'{key} has invalid shape, dtype or values: {value.shape}, {value.dtype}')
    combined.update(add_proximity(combined['AudioFeatures'], combined['TextFeatures']))
    for key in ('Cosine_Sim', 'Cosine_Ang', 'L1', 'L2'):
        if combined[key].shape != (n, 1) or not np.isfinite(combined[key]).all():
            raise ValueError(f'{key} is invalid')
    return combined


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **arrays)
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', encoding='utf-8', delete=False) as file:
        temporary = Path(file.name)
        json.dump(data, file, indent=2)
    os.replace(temporary, path)


def load_cache(path, entry, fingerprint, expected_names, expected_labels, feature_dim):
    path = Path(path)
    if not path.is_file() or not entry or entry.get('status') != 'complete' or entry.get('fingerprint') != fingerprint:
        return None
    if entry.get('dataset_source') != 'qnv8071/xacle-2027-data':
        return None
    if entry.get('sample_count') != len(expected_names) or entry.get('feature_dim') != feature_dim:
        return None
    if entry.get('indices') != list(range(len(expected_names))) or entry.get('sha256') != file_sha256(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(ARRAY_KEYS) - set(loaded.files):
                return None
            arrays = {key: loaded[key] for key in ARRAY_KEYS}
        if not np.array_equal(arrays['indices'], np.arange(len(expected_names))):
            return None
        if not np.array_equal(arrays['wav_file_name'].astype(str), np.asarray(expected_names).astype(str)):
            return None
        if not np.allclose(arrays['labels'], expected_labels, atol=1e-6, rtol=0):
            return None
        n = len(expected_names)
        if any(arrays[k].shape != (n, feature_dim) or arrays[k].dtype.kind != 'f' for k in ('AudioFeatures', 'TextFeatures')):
            return None
        if any(arrays[k].shape != (n, 1) or arrays[k].dtype.kind != 'f' for k in ('Cosine_Sim', 'Cosine_Ang', 'L1', 'L2')):
            return None
        if any(not np.isfinite(arrays[k]).all() for k in ARRAY_KEYS if arrays[k].dtype.kind == 'f'):
            return None
        if any(str(arrays[k].dtype) != entry.get('dtypes', {}).get(k) for k in ARRAY_KEYS):
            return None
        return arrays
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def fingerprint(cfg, split, model, names, labels, sources, versions, code_hashes, wav_identity):
    checkpoint = cfg['model']['M2D']['ckpt'] if model == 'M2D_Clap' else cfg['model']['MGA']['ckpt'] if model == 'MGA_Clap' else None
    identity = ({'path': checkpoint, 'bytes': Path(checkpoint).stat().st_size,
                 'mtime_ns': Path(checkpoint).stat().st_mtime_ns} if checkpoint else {'model': model})
    value = {'dataset': 'qnv8071/xacle-2027-data', 'split': split, 'model': model,
             'names': np.asarray(names).tolist(), 'labels': np.asarray(labels).tolist(),
             'metadata_sha256': file_sha256(cfg[f'{split}_list']),
             'wav_manifest_sha256': wav_identity,
             'batch_size': BATCH_SIZES[model], 'max_len': cfg['max_len'], 'sample_rate': cfg['sample_rate'],
             'checkpoint': identity, 'sources': sources, 'versions': versions, 'code_hashes': code_hashes}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), identity


def device_for_worker(gpu_id):
    if gpu_id not in (0, 1):
        raise ValueError('GPU worker id must be 0 or 1')
    return f'cuda:{gpu_id}'


def extract_worker(gpu_id, indices, split, model_name, cfg, shard_path, stats_path):
    """CUDA is first queried in this spawned child, never in the parent."""
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        from datasets.fetch_data import get_dataset
        from features import extract_features
        from load_pretrained_models.load_model import load_model
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError(f'Expected 2 CUDA GPUs, found {torch.cuda.device_count()}')
        torch.cuda.set_device(gpu_id)
        for device_id in (0, 1):
            properties = torch.cuda.get_device_properties(device_id)
            log(f'GPU {device_id}: {properties.name}, {properties.total_memory} bytes')
        worker_cfg = dict(cfg, device=device_for_worker(gpu_id))
        torch.cuda.reset_peak_memory_stats(gpu_id)
        stats = {'gpu_id': gpu_id, 'device': worker_cfg['device'], 'split': split, 'model': model_name,
                 'batch_size': BATCH_SIZES[model_name], 'allocated_before_load': torch.cuda.memory_allocated(gpu_id)}
        dataset = get_dataset(cfg[f'{split}_list'], str(Path(cfg['wav_dir']) / split), cfg['max_len'], cfg['sample_rate'])
        loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZES[model_name], shuffle=False,
                            num_workers=0, collate_fn=dataset.collate_fn)
        loaded = load_model(model_name, worker_cfg)
        if model_name == 'MS_Clap':
            loaded[0].clap.to(worker_cfg['device'])
        elif model_name == 'Laion_Clap':
            loaded[1].to(worker_cfg['device'])
            loaded[3].to(worker_cfg['device'])
        elif model_name == 'Whisper':
            loaded[2].to(worker_cfg['device'])
        else:
            loaded[0].to(worker_cfg['device'])
        stats['allocated_after_load'] = torch.cuda.memory_allocated(gpu_id)
        if model_name == 'M2D_Clap':
            features, labels = extract_features.M2DCLAP(loader, loaded[0], worker_cfg)
        elif model_name == 'MS_Clap':
            features, labels = extract_features.MSCLAP(loader, loaded[0], worker_cfg)
        elif model_name == 'Laion_Clap':
            features, labels = extract_features.LaionCLAP(loader, *loaded[:4], worker_cfg)
        elif model_name == 'MGA_Clap':
            features, labels = extract_features.MGAClap(loader, loaded[0], worker_cfg)
        elif model_name == 'Whisper':
            features, labels = extract_features.Whisper(loader, *loaded[:3], worker_cfg)
        else:
            raise ValueError(model_name)
        stats['parameters'] = loaded[-1]
        stats['peak_allocated'] = torch.cuda.max_memory_allocated(gpu_id)
        stats['peak_reserved'] = torch.cuda.max_memory_reserved(gpu_id)
        names = np.asarray(dataset.df.iloc[indices]['wav_file_name'].astype(str).tolist(), dtype=str)
        arrays = {'indices': np.asarray(indices, dtype=np.int64), 'wav_file_name': names,
                  'labels': np.asarray(labels), **features}
        if len(arrays['AudioFeatures']) != len(indices):
            raise ValueError('Extractor output length differs from shard indices')
        atomic_npz(shard_path, arrays)
        del loaded, loader, dataset, features, labels, arrays
        gc.collect()
        torch.cuda.empty_cache()
        stats['allocated_after_cleanup'] = torch.cuda.memory_allocated(gpu_id)
        stats['reserved_after_cleanup'] = torch.cuda.memory_reserved(gpu_id)
        atomic_json(stats_path, stats)
        log(f'{split}/{model_name} GPU {gpu_id} complete: {len(indices)} rows; peak={stats["peak_allocated"]} bytes')
    except Exception:
        atomic_json(str(stats_path) + '.error.json', {'gpu_id': gpu_id, 'traceback': traceback.format_exc()})
        raise


def run_workers(split, model, cfg, names, shard_dir):
    context = mp.get_context('spawn')
    partitions = partition_indices(len(names), BATCH_SIZES[model])
    processes = []
    for gpu_id, indices in enumerate(partitions):
        shard = shard_dir / f'gpu{gpu_id}.npz'
        stats = shard_dir / f'gpu{gpu_id}.json'
        process = context.Process(target=extract_worker, args=(gpu_id, indices, split, model, cfg, shard, stats))
        process.start()
        processes.append(process)
    while any(process.is_alive() for process in processes):
        for process in processes:
            process.join(timeout=1)
        if any(process.exitcode not in (None, 0) for process in processes):
            for process in processes:
                if process.is_alive():
                    process.terminate()
            break
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        errors = [Path(str(shard_dir / f'gpu{i}.json') + '.error.json').read_text()
                  for i, process in enumerate(processes) if process.exitcode and Path(str(shard_dir / f'gpu{i}.json') + '.error.json').exists()]
        raise RuntimeError(f'{split}/{model} worker failed: {errors}')
    with np.load(shard_dir / 'gpu0.npz', allow_pickle=False) as first, np.load(shard_dir / 'gpu1.npz', allow_pickle=False) as second:
        shards = [{key: file[key] for key in ('indices', 'wav_file_name', 'labels', 'AudioFeatures', 'TextFeatures')}
                  for file in (first, second)]
    stats = [json.loads((shard_dir / f'gpu{i}.json').read_text()) for i in (0, 1)]
    return shards, stats
