import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kaggle import dual_gpu


class DualGpuTests(unittest.TestCase):
    def test_batches_and_balanced_ordered_shards(self):
        self.assertEqual(dual_gpu.BATCH_SIZES['Whisper'], 1)
        self.assertTrue(all(dual_gpu.BATCH_SIZES[m] == 8 for m in dual_gpu.MODELS if m != 'Whisper'))
        a, b = dual_gpu.partition_indices(17, 8)
        self.assertEqual(a + b, list(range(17)))
        self.assertEqual(len(set(a) & set(b)), 0)
        self.assertLessEqual(abs(len(a)-len(b)), 8)
        self.assertEqual(len(a) % 8, 0)

    def test_merge_restores_order_and_global_proximity(self):
        names = np.array(['a.wav', 'b.wav', 'c.wav', 'd.wav'])
        labels = np.array([-.5, 0, .5, 1], dtype=np.float32)
        raw = []
        for indices in ([0, 1], [2, 3]):
            raw.append({'indices': np.array(indices), 'wav_file_name': names[indices],
                        'labels': labels[indices],
                        'AudioFeatures': np.array([[i+1, i+2] for i in indices], dtype=np.float32),
                        'TextFeatures': np.array([[i+2, i+1] for i in indices], dtype=np.float32)})
        merged = dual_gpu.merge_shards(list(reversed(raw)), names, labels, 2)
        self.assertEqual(merged['indices'].tolist(), [0, 1, 2, 3])
        self.assertEqual(merged['wav_file_name'].tolist(), names.tolist())
        self.assertEqual(merged['L1'].shape, (4, 1))
        self.assertEqual(merged['L1'].max(), 1)
        with self.assertRaisesRegex(ValueError, 'duplicate|missing'):
            dual_gpu.merge_shards([raw[0], raw[0]], names, labels, 2)

    def test_cache_requires_manifest_fingerprint_and_valid_arrays(self):
        names = np.array(['a.wav', 'b.wav'])
        labels = np.array([0, .2], dtype=np.float32)
        shards = [{'indices': np.array([i]), 'wav_file_name': names[i:i+1], 'labels': labels[i:i+1],
                   'AudioFeatures': np.array([[1+i, 2+i]], dtype=np.float32),
                   'TextFeatures': np.array([[2+i, 1+i]], dtype=np.float32)} for i in (0, 1)]
        arrays = dual_gpu.merge_shards(shards, names, labels, 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'features.npz'
            digest = dual_gpu.atomic_npz(path, arrays)
            entry = {'status': 'complete', 'fingerprint': 'correct', 'sha256': digest,
                     'dataset_source': 'qnv8071/xacle-2027-data',
                     'sample_count': 2, 'feature_dim': 2, 'indices': [0, 1],
                     'dtypes': {k: str(v.dtype) for k, v in arrays.items()}}
            self.assertIsNotNone(dual_gpu.load_cache(path, entry, 'correct', names, labels, 2))
            self.assertIsNone(dual_gpu.load_cache(path, entry, 'wrong', names, labels, 2))
            entry['status'] = 'incomplete'
            self.assertIsNone(dual_gpu.load_cache(path, entry, 'correct', names, labels, 2))
            entry['status'] = 'complete'
            entry['dataset_source'] = 'wrong'
            self.assertIsNone(dual_gpu.load_cache(path, entry, 'correct', names, labels, 2))
            entry['dataset_source'] = 'qnv8071/xacle-2027-data'
            self.assertIsNone(dual_gpu.load_cache(path, entry, 'correct', names[::-1], labels, 2))
            self.assertIsNone(dual_gpu.load_cache(path, entry, 'correct', names, labels, 3))

    def test_spawned_workers_get_distinct_indices_and_devices(self):
        events = []
        class FakeProcess:
            def __init__(self, target, args):
                self.args = args
                self.exitcode = None
            def start(self):
                gpu, indices, split, model, cfg, shard, stats = self.args
                events.append(('start', gpu, indices))
                dual_gpu.atomic_npz(shard, {'indices': np.asarray(indices),
                    'wav_file_name': np.asarray([f'{i}.wav' for i in indices]),
                    'labels': np.asarray(indices, dtype=np.float32),
                    'AudioFeatures': np.ones((len(indices), 2), dtype=np.float32),
                    'TextFeatures': np.ones((len(indices), 2), dtype=np.float32)})
                dual_gpu.atomic_json(stats, {'parameters': 2, 'gpu_id': gpu})
                self.exitcode = 0
            def is_alive(self):
                return False
            def join(self, timeout=None):
                events.append(('join', self.args[0]))
        context = type('Context', (), {'Process': FakeProcess})()
        with tempfile.TemporaryDirectory() as temp, patch.object(dual_gpu.mp, 'get_context', return_value=context):
            shards, stats = dual_gpu.run_workers('train', 'Whisper', {}, list(range(4)), Path(temp))
        self.assertEqual([x[0] for x in events[:2]], ['start', 'start'])
        self.assertEqual(events[0][2], [0, 1])
        self.assertEqual(events[1][2], [2, 3])
        self.assertEqual([s['gpu_id'] for s in stats], [0, 1])
        self.assertEqual(dual_gpu.device_for_worker(0), 'cuda:0')
        self.assertEqual(dual_gpu.device_for_worker(1), 'cuda:1')


if __name__ == '__main__':
    unittest.main()
