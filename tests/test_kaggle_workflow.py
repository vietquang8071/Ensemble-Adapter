import json
import ast
import base64
import io
import re
import tarfile
import tempfile
import unittest
from pathlib import Path

from kaggle.workflow import validate_config, stage_kernel, verify_outputs, push_command
from kaggle import run_xacle


ROOT = Path(__file__).resolve().parents[1]


class KaggleWorkflowTests(unittest.TestCase):
    def test_config_preserves_reference_parameters_and_mounts(self):
        reference = json.loads((ROOT / 'config_submission1.json').read_text())
        config = json.loads((ROOT / 'config_kaggle.json').read_text())
        self.assertEqual(config['SVR'], reference['SVR'])
        self.assertEqual(config['batch_size'], reference['batch_size'])
        self.assertEqual(config['device'], 'cuda')
        self.assertTrue(config['wav_dir'].endswith('/wav'))
        self.assertTrue(config['model']['M2D']['ckpt'].endswith('/checkpoint-30.pth'))
        self.assertTrue(config['test_average_list'].endswith('/meta_data/test_average.csv'))
        validate_config(config)

    def test_rejects_read_only_output(self):
        config = json.loads((ROOT / 'config_kaggle.json').read_text())
        config['output_dir'] = '/kaggle/input/outputs'
        with self.assertRaisesRegex(ValueError, 'output_dir'):
            validate_config(config)

    def test_stage_includes_local_research_code(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / 'kernel'
            stage_kernel(ROOT, stage, 'preflight')
            self.assertTrue((stage / 'features' / 'extract_features.py').exists())
            self.assertEqual(json.loads((stage / 'mode.json').read_text())['mode'], 'preflight')
            metadata = json.loads((stage / 'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private'])
            self.assertTrue(metadata['enable_gpu'])
            self.assertEqual(len(metadata['dataset_sources']), 2)
            script = (stage / 'run_xacle.py').read_text()
            ast.parse(script)
            encoded = re.search(r"PAYLOAD = '([A-Za-z0-9+/=]+)'", script).group(1)
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)), mode='r:gz') as bundle:
                self.assertIn('features/extract_features.py', bundle.getnames())
                self.assertIn('train.py', bundle.getnames())
                self.assertIn('config_kaggle.json', bundle.getnames())
                self.assertIn('dual_gpu.py', bundle.getnames())
                self.assertIn('dual_gpu_pipeline.py', bundle.getnames())

    def test_download_verification_rejects_incomplete_training(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'outputs/version_config_kaggle'
            output.mkdir(parents=True)
            (output / 'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'config.json'):
                verify_outputs(Path(temp))

    def test_runtime_cleanup_keeps_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            for name in ('xacle-env', 'xacle-src', 'm2d', 'MGA-CLAP', 'model_links', 'outputs'):
                (work / name).mkdir()
            (work / 'outputs/preflight_report.json').write_text('{}')
            run_xacle.cleanup(work)
            self.assertTrue((work / 'outputs/preflight_report.json').exists())
            self.assertFalse((work / 'xacle-env').exists())

    def test_push_uses_dual_t4_without_quick_save(self):
        command = push_command(Path('/tmp/kernel'))
        self.assertEqual(command[:3], ['kaggle', 'kernels', 'push'])
        self.assertEqual(command[-2:], ['--accelerator', 'NvidiaTeslaT4'])
        self.assertNotIn('--no-run', command)


if __name__ == '__main__':
    unittest.main()
