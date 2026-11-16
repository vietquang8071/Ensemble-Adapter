"""Exercise the original SVR code on small cached feature arrays, without a GPU."""
import ast
import gc
import json
import os
import pickle
import tempfile
import types
import unittest
from math import sqrt
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from kaggle.dual_gpu import add_proximity

ROOT = Path(__file__).resolve().parents[1]


def functions_from(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)


class CachedSvrTests(unittest.TestCase):
    def test_original_fit_and_artifacts(self):
        try:
            import pandas as pd
            from sklearn.preprocessing import Normalizer, StandardScaler
            from sklearn.svm import SVR
            from scipy.optimize import minimize
        except ImportError as exc:
            self.skipTest(f'Optional isolated sklearn test dependencies absent: {exc}')
        namespace = {'np': np, 'pd': pd, 'os': os, 'json': json, 'gc': gc,
                     'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
                     'Normalizer': Normalizer, 'StandardScaler': StandardScaler,
                     'SVR': SVR, 'minimize': minimize, 'sqrt': sqrt,
                     'Iterable': Iterable, 'List': List, 'Tuple': Tuple, 'pickle': pickle}
        functions_from(ROOT / 'evaluate.py',
                       {'_check_same_length', '_mean', '_ranks', 'lcc_pearson', 'srcc_spearman',
                        '_tie_pairs_count', 'ktau_b', 'mse', 'evaluate_all'}, namespace)
        functions_from(ROOT / 'utils/utils.py',
                       {'Save_SKLearnPreprocessors', 'Save_SVR', 'Save_Preprocessor_and_SVR'}, namespace)
        namespace['utils'] = types.SimpleNamespace(Save_Preprocessor_and_SVR=namespace['Save_Preprocessor_and_SVR'])
        functions_from(ROOT / 'train.py',
                       {'GetLabels', 'Fit_SVR', 'Fit_SVR_A', 'Fit_SVR_B', 'RMSE', 'Param_count',
                        'fit_from_features'}, namespace)
        with tempfile.TemporaryDirectory() as temp:
            cfg = json.loads((ROOT / 'config_submission1.json').read_text())
            cfg['output_dir'] = temp
            cfg['config_filename'] = 'config_kaggle.json'
            feature_sets, label_sets = {}, {}
            for split, count in (('train', 8), ('validation', 4)):
                csv_path = Path(temp) / f'{split}.csv'
                pd.DataFrame({'wav_file_name': [f'{split}-{i}.wav' for i in range(count)]}).to_csv(csv_path, index=False)
                cfg[f'{split}_list'] = str(csv_path)
                labels = np.linspace(-.7, .7, count, dtype=np.float32)
                feature_sets[split], label_sets[split] = {}, {}
                for j, model in enumerate(('M2D_Clap', 'MS_Clap', 'Laion_Clap', 'MGA_Clap', 'Whisper')):
                    audio = np.stack((np.arange(count) + j + 1, np.arange(count) * .3 + 2), axis=1).astype(np.float32)
                    text = np.stack((np.arange(count) * .6 + 1, np.arange(count) + j + 3), axis=1).astype(np.float32)
                    feature_sets[split][model] = {'AudioFeatures': audio, 'TextFeatures': text,
                                                  **add_proximity(audio, text)}
                    label_sets[split][model] = labels
            result = namespace['fit_from_features'](feature_sets['train'], feature_sets['validation'],
                                                    label_sets['train'], label_sets['validation'],
                                                    {name: 10 for name in feature_sets['train']}, cfg)
            artifact = Path(result['artifact_dir'])
            self.assertEqual(len(result['weights']), 2)
            self.assertEqual(result['train']['N'], 8)
            for group in ('A', 'B'):
                for name in (f'SVR_{group}', f'Normalizer_{group}', f'StandardScaler_{group}'):
                    self.assertTrue((artifact / group / f'{name}.pkl').is_file())
            for name in ('config.json', 'metrics.json', 'train_predictions.csv', 'validation_predictions.csv'):
                self.assertTrue((artifact / name).is_file())


if __name__ == '__main__':
    unittest.main()
