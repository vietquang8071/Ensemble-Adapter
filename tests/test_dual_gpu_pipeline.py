import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kaggle.workflow import stage_kernel

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_extraction_cache_reuse_and_pending_publication(self):
        try:
            import pandas as pd
            sys.path.insert(0, str(ROOT / 'kaggle'))
            import dual_gpu_pipeline as pipeline
            from dual_gpu import DIMS, partition_indices
        except ImportError as exc:
            self.skipTest(f'Optional isolated pandas test dependency absent: {exc}')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            stage = base / 'stage'
            stage_kernel(ROOT, stage, 'train')
            cfg = json.loads((ROOT / 'config_kaggle.json').read_text())
            cfg['output_dir'] = str(base / 'outputs')
            for split, count in (('train', 4), ('validation', 2), ('test_average', 2)):
                csv_path = base / f'{split}.csv'
                pd.DataFrame({'wav_file_name': [f'{split}-{i}.wav' for i in range(count)],
                              'average_score': np.linspace(2, 8, count)}).to_csv(csv_path, index=False)
                cfg[f'{split}_list'] = str(csv_path)
            for model in ('M2D', 'MGA'):
                checkpoint = base / f'{model}.pt'
                checkpoint.write_bytes(b'test')
                cfg['model'][model]['ckpt'] = str(checkpoint)
            calls = []
            def fake_workers(split, model, config, names, shard_dir):
                calls.append((split, model))
                result = []
                for gpu, indices in enumerate(partition_indices(len(names), 1)):
                    audio = np.zeros((len(indices), DIMS[model]), dtype=np.float32)
                    text = np.zeros_like(audio)
                    audio[:, :2] = [gpu + 1, 2]
                    text[:, :2] = [2, gpu + 3]
                    labels = ((pd.read_csv(config[f'{split}_list'])['average_score'].to_numpy(dtype=np.float32) - 5) / 5)
                    result.append({'indices': np.asarray(indices), 'wav_file_name': names[indices],
                                   'labels': labels[indices], 'AudioFeatures': audio, 'TextFeatures': text})
                return result, [{'gpu_id': i, 'parameters': 10, 'peak_allocated': 100} for i in (0, 1)]
            def fake_fit(train_features, val_features, train_labels, val_labels, parameters, config):
                artifact = Path(config['output_dir']) / 'version_config_kaggle'
                for group in ('A', 'B'):
                    (artifact / group).mkdir(parents=True, exist_ok=True)
                    for name in (f'SVR_{group}', f'Normalizer_{group}', f'StandardScaler_{group}'):
                        (artifact / group / f'{name}.pkl').write_bytes(b'test')
                config['W'] = [0.5, 0.5]
                return {'train': {'N': 4}, 'validation': {'N': 2}, 'weights': [0.5, 0.5],
                        'artifact_dir': str(artifact)}
            def fake_predict(features, config):
                return np.array([4.0, 6.0], dtype=np.float32)
            with patch.object(pipeline, 'check_dataset', return_value={
                'train': {'wav_manifest_sha256': 'train-hash'},
                'validation': {'wav_manifest_sha256': 'val-hash'},
                'test_average': {'wav_manifest_sha256': 'test-hash'}}), \
                 patch.object(pipeline, 'attached_feature_root', return_value=None), \
                 patch.object(pipeline, 'run_workers', side_effect=fake_workers), \
                 patch.object(pipeline.importlib.metadata, 'version', return_value='test'), \
                 patch.dict(sys.modules, {'train': types.SimpleNamespace(fit_from_features=fake_fit)}), \
                 patch.object(pipeline, 'predict_test', side_effect=fake_predict), \
                 patch.object(pipeline, 'evaluate_test', return_value={'N': 2, 'MSE': 0.1}):
                first = pipeline.run(cfg, base / 'outputs', {'m2d': 'sha1', 'mga': 'sha2'}, stage)
                self.assertEqual(len(calls), 15)
                self.assertEqual(first['publication']['status'], 'pending')
                self.assertEqual(first['svr_training'], 'complete')
                self.assertEqual(first['test_inference'], 'complete')
                self.assertTrue((base / 'outputs/version_config_kaggle/test_predictions.csv').exists())
                self.assertEqual(first['metrics']['test']['N'], 2)
                self.assertTrue(cfg['test_list'].endswith('/meta_data/test.csv'))
                second = pipeline.run(cfg, base / 'outputs', {'m2d': 'sha1', 'mga': 'sha2'}, stage)
                self.assertEqual(len(calls), 15)
                self.assertEqual(second['extraction'], 'complete')

    def test_test_prediction_keeps_original_weighting_and_clipping(self):
        try:
            sys.path.insert(0, str(ROOT / 'kaggle'))
            import dual_gpu_pipeline as pipeline
        except ImportError as exc:
            self.skipTest(f'Optional isolated pandas test dependency absent: {exc}')
        fake_inference = types.SimpleNamespace(
            Predict_SVR_A=lambda split, features, cfg: np.array([-10.0, 8.0]),
            Predict_SVR_B=lambda split, features, cfg: np.array([0.0, 20.0]))
        with patch.dict(sys.modules, {'inference': fake_inference}):
            result = pipeline.predict_test({}, {'W': [0.5, 0.5]})
        self.assertEqual(result.tolist(), [0.0, 10.0])


if __name__ == '__main__':
    unittest.main()
