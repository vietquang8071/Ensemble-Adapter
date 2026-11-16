"""Cache-backed dual-T4 extraction followed by the original SVR fit."""
import importlib.metadata
import json
import os
import shutil
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from dual_gpu import (MODELS, SLUGS, DIMS, BATCH_SIZES, atomic_json, atomic_npz,
                      file_sha256, fingerprint, load_cache, log, merge_shards, run_workers)
from model_preflight import check_dataset

DATASET = 'qnv8071/xacle-extracted-features'


def attached_feature_root():
    base = Path('/kaggle/input/datasets/qnv8071/xacle-extracted-features')
    for root in (base / 'outputs/features', base / 'features', base):
        if (root / 'manifest.json').is_file():
            return root
    return None


def copy_cache(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.copying')
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def predict_test(features, cfg):
    """Reuse the original SVR feature order, preprocessing, and denormalization."""
    from inference import Predict_SVR_A, Predict_SVR_B
    a = Predict_SVR_A('test', features, cfg)
    b = Predict_SVR_B('test', features, cfg)
    return np.clip(cfg['W'][0] * a + cfg['W'][1] * b, 0, 10)


def evaluate_test(predictions, ground_truth):
    from evaluate import evaluate_all
    return evaluate_all(predictions, ground_truth)


def run(cfg, output_dir, sources, root):
    """Parent process never queries CUDA before spawning extractor workers."""
    output_dir = Path(output_dir)
    feature_root = output_dir / 'features'
    feature_root.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'execution_report.json'
    manifest_path = feature_root / 'manifest.json'
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'torchaudio', 'numpy', 'transformers', 'msclap', 'timm', 'scikit-learn')}
    code_hashes = {name: file_sha256(Path(root) / name) for name in
                   ('features/extract_features.py', 'features/proximity_features.py',
                    'datasets/fetch_data.py', 'load_pretrained_models/load_model.py', 'dual_gpu.py')}
    report = {'extraction': 'started', 'publication': {'dataset': DATASET, 'status': 'pending',
              'reason': 'No configured Kaggle API secret; create a private Dataset from the completed Notebook Output'},
              'svr_training': 'pending', 'test_inference': 'pending',
              'gpu': {}, 'artifacts': {}, 'metrics': None}
    manifest = {'schema_version': 1, 'dataset_source': 'qnv8071/xacle-2027-data',
                'software_versions': versions, 'source_commits': sources, 'code_hashes': code_hashes,
                'entries': {}}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get('schema_version') == 1 and previous.get('dataset_source') == manifest['dataset_source']:
            manifest['entries'] = previous.get('entries', {})
    atomic_json(report_path, report)
    atomic_json(manifest_path, manifest)
    try:
        report['dataset'] = check_dataset(cfg)
        extract_cfg = dict(cfg, test_list=cfg['test_average_list'])
        frames = {split: pd.read_csv(extract_cfg[f'{split}_list']) for split in ('train', 'validation', 'test')}
        names = {split: np.asarray(frame['wav_file_name'].astype(str).tolist(), dtype=str) for split, frame in frames.items()}
        labels = {split: ((frame['average_score'].to_numpy(dtype=np.float32) - 5) / 5).astype(np.float32)
                  for split, frame in frames.items()}
        attached_root = attached_feature_root()
        attached_manifest = json.loads((attached_root / 'manifest.json').read_text()) if attached_root else {}
        parameter_counts = {}
        for model in MODELS:
            for split in ('train', 'validation', 'test'):
                slug = SLUGS[model]
                key = f'{split}/{slug}'
                path = feature_root / split / f'{slug}.npz'
                fp, checkpoint = fingerprint(extract_cfg, split, model, names[split], labels[split], sources,
                                             versions, code_hashes, report['dataset']['test_average' if split == 'test' else split]['wav_manifest_sha256'])
                entry = manifest['entries'].get(key)
                arrays = load_cache(path, entry, fp, names[split], labels[split], DIMS[model])
                if arrays is None and attached_root:
                    source_path = attached_root / split / f'{slug}.npz'
                    entry = attached_manifest.get('entries', {}).get(key)
                    arrays = load_cache(source_path, entry, fp, names[split], labels[split], DIMS[model])
                    if arrays is not None:
                        copy_cache(source_path, path)
                        log(f'Reused attached cache: {key}')
                stats = entry.get('gpu_stats', []) if entry and arrays is not None else []
                if arrays is None:
                    log(f'Extracting {key} on both T4 GPUs; batch={BATCH_SIZES[model]} per GPU')
                    shard_dir = feature_root / '.shards' / split / slug
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    shards, stats = run_workers(split, model, extract_cfg, names[split], shard_dir)
                    arrays = merge_shards(shards, names[split], labels[split], DIMS[model])
                    digest = atomic_npz(path, arrays)
                    parameter_counts[model] = stats[0]['parameters']
                    del shards
                else:
                    digest = file_sha256(path)
                    parameter_counts[model] = entry.get('parameters', 0)
                entry = {'status': 'complete', 'dataset_source': manifest['dataset_source'], 'split': split,
                         'model': model, 'checkpoint': checkpoint, 'fingerprint': fp, 'sha256': digest,
                         'sample_count': len(names[split]), 'indices': list(range(len(names[split]))),
                         'feature_dim': DIMS[model], 'dtypes': {k: str(v.dtype) for k, v in arrays.items()},
                         'batch_size_per_gpu': BATCH_SIZES[model], 'parameters': parameter_counts[model],
                         'gpu_stats': stats, 'source_commits': sources, 'software_versions': versions}
                manifest['entries'][key] = entry
                atomic_json(manifest_path, manifest)
                report['gpu'][key] = stats
                report['artifacts'][key] = str(path)
                atomic_json(report_path, report)
                del arrays
        if len(manifest['entries']) != 15:
            raise RuntimeError('Feature manifest is incomplete')
        report['extraction'] = 'complete'
        log('Feature extraction complete; private Dataset publication pending; training SVR from validated local caches')
        atomic_json(report_path, report)
        feature_sets = {}
        label_sets = {}
        for split in ('train', 'validation', 'test'):
            feature_sets[split] = {}
            label_sets[split] = {}
            for model in MODELS:
                key = f'{split}/{SLUGS[model]}'
                arrays = load_cache(feature_root / split / f'{SLUGS[model]}.npz',
                                    manifest['entries'][key], manifest['entries'][key]['fingerprint'],
                                    names[split], labels[split], DIMS[model])
                if arrays is None:
                    raise RuntimeError(f'Final cache validation failed: {key}')
                feature_sets[split][model] = {k: arrays[k] for k in
                                              ('AudioFeatures', 'TextFeatures', 'Cosine_Sim', 'Cosine_Ang', 'L1', 'L2')}
                label_sets[split][model] = arrays['labels']
        from train import fit_from_features
        cfg['config_filename'] = 'config_kaggle.json'
        metrics = fit_from_features(feature_sets['train'], feature_sets['validation'],
                                    label_sets['train'], label_sets['validation'], parameter_counts, cfg)
        report['svr_training'] = 'complete'
        report['metrics'] = {k: metrics[k] for k in ('train', 'validation', 'weights')}
        artifact_dir = Path(metrics['artifact_dir'])
        report['artifacts']['svr'] = {name: str(artifact_dir / group / f'{name}.pkl')
                                      for group, names_in_group in (('A', ('SVR_A', 'Normalizer_A', 'StandardScaler_A')),
                                                                    ('B', ('SVR_B', 'Normalizer_B', 'StandardScaler_B')))
                                      for name in names_in_group}
        for path in report['artifacts']['svr'].values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        test_predictions = predict_test(feature_sets['test'], cfg)
        test_reference = frames['test']['average_score'].to_numpy(dtype=float)
        if len(test_predictions) != len(test_reference):
            raise ValueError('Test predictions do not align with test_average.csv')
        test_metrics = evaluate_test(test_predictions, test_reference)
        report['metrics']['test'] = test_metrics
        pd.DataFrame({'wav_file_name': names['test'], 'pred_score': test_predictions,
                      'average_score': test_reference}).to_csv(artifact_dir / 'test_predictions.csv', index=False)
        pd.DataFrame({'wav_file_name': names['test'],
                      'prediction_score': np.round(test_predictions, 2)}).to_csv(
                          artifact_dir / 'inference_result_for_test.csv', index=False)
        atomic_json(artifact_dir / 'test_metrics.json', test_metrics)
        report['artifacts']['test_predictions'] = str(artifact_dir / 'test_predictions.csv')
        report['artifacts']['test_metrics'] = str(artifact_dir / 'test_metrics.json')
        (output_dir / 'models').mkdir(exist_ok=True)
        atomic_json(output_dir / 'models/index.json', report['artifacts']['svr'])
        (output_dir / 'metrics').mkdir(exist_ok=True)
        atomic_json(output_dir / 'metrics/summary.json', report['metrics'])
        report['test_inference'] = 'complete'
        atomic_json(report_path, report)
        return report
    except Exception:
        log('Pipeline failed; see execution_report.json for traceback')
        report['error'] = traceback.format_exc()
        if report['extraction'] != 'complete':
            report['extraction'] = 'failed'
        if report['svr_training'] != 'complete':
            report['svr_training'] = 'failed' if report['extraction'] == 'complete' else 'not_started'
        if report['test_inference'] != 'complete':
            report['test_inference'] = 'failed' if report['svr_training'] == 'complete' else 'not_started'
        atomic_json(report_path, report)
        raise
