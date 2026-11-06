"""Check native launcher restart decisions without starting GPU processes."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training.run_jd import train_to_completion


class RestartTests(unittest.TestCase):
    def test_resumes_from_new_checkpoint_and_finishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = SimpleNamespace(output_dir=output, evaluate_only=False, resume=False,
                                   max_stalls=2, max_restarts=3, retry_delay=0, gpus=['0'] * 8,
                                   data_dir=output, tokenizer_dir=output, eval_cache_dir=output)
            settings = {'training': {'max_steps': 20}}
            calls = []

            def run(command, path, environment):
                calls.append(command)
                (output / 'train' / f'step-{10 * len(calls)}.pt').write_bytes(b'complete')
                return 1 if len(calls) == 1 else 0

            with patch('training.run_jd.run_stage', side_effect=run):
                final = train_to_completion(args, settings, {})
            self.assertEqual(final.name, 'step-20.pt')
            self.assertEqual(len(calls), 2)
            self.assertNotIn('--resume', calls[0])
            self.assertEqual(calls[1][calls[1].index('--resume') + 1],
                             str(output / 'train' / 'step-10.pt'))

    def test_stops_when_retries_make_no_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = SimpleNamespace(output_dir=output, evaluate_only=False, resume=False,
                                   max_stalls=2, max_restarts=5, retry_delay=0, gpus=['0'] * 8,
                                   data_dir=output, tokenizer_dir=output, eval_cache_dir=output)
            calls = []

            def run(command, path, environment):
                calls.append(command)
                (output / 'train' / 'step-10.pt').write_bytes(b'complete')
                return 1

            with patch('training.run_jd.run_stage', side_effect=run):
                with self.assertRaisesRegex(RuntimeError, 'no progress'):
                    train_to_completion(args, {'training': {'max_steps': 20}}, {})
            self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
