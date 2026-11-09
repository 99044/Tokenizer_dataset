"""Check that final C_i requires a complete, consistent observed training path."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from qwen3_moe.plan import make_recipe
from training.audit_qwen3_moe_flops import audit
from context_curriculum import curriculum_completed_flops, curriculum_stage


class TrainingPathAuditTests(unittest.TestCase):
    def test_complete_path_produces_actual_flops_and_incomplete_or_changed_paths_fail(self):
        _, settings, _ = make_recipe("3e17")
        rows = []
        for step in range(1, settings["training"]["max_steps"] + 1):
            stage = curriculum_stage(settings, step)
            rows.append(dict(step=step, **{
                "train/context_length": stage["length"], "train/micro_batch_size": stage["micro_batch_size"],
                "train/grad_accum_steps": stage["grad_accum_steps"], "train/curriculum_stage_index": stage["stage_index"],
                "train/tokens": step * stage["tokens_per_update"],
                "train/completed_flops": curriculum_completed_flops(settings, step),
            }))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            def write(records):
                path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            write(rows)
            result = audit(settings, path)
            self.assertEqual(result["actual_training_flops"], settings["budget"]["planned_training_flops"])
            self.assertEqual(result["completed_optimizer_steps"], len(rows))
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit(settings, path)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit(settings, path, allow_time_limit=True)
            limited = [dict(row) for row in rows[:2]]
            limited[-1]["train/stop_reason"] = "time_limit"
            write(limited)
            result = audit(settings, path, allow_time_limit=True)
            self.assertEqual(result["status"], "passed_time_limited_prefix_audit")
            self.assertEqual(result["actual_training_flops"], limited[-1]["train/completed_flops"])
            self.assertEqual(result["final_context_length"], 2048)
            self.assertFalse(result["within_one_percent"])
            broken = [dict(row) for row in rows]
            broken[-1]["train/context_length"] = 2048
            write(broken)
            with self.assertRaisesRegex(ValueError, "runtime shape"):
                audit(settings, path)
            write([rows[0], *rows])
            with self.assertRaisesRegex(ValueError, "gaps or duplicates"):
                audit(settings, path)


if __name__ == "__main__":
    unittest.main()
