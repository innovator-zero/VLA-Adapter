"""Standard-library regression tests; no GPU, checkpoint or simulator required."""
import ast
from collections import deque
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run_libero_plus_eval as runner

reports = runner.reports


class PlusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.suite = "libero_spatial"
        self.metadata = {
            suite: {f"task_{i}": {"id": i, "category": "lighting"} for i in range(2)}
            for suite in reports.LIBERO_PLUS_SUITES
        }
        self.options = SimpleNamespace(output_dir=self.output, save_name="test", resume=False)
        self.cfg = SimpleNamespace(
            task_suite_name=self.suite, num_trials_per_task=1, num_open_loop_steps=8,
            num_steps_wait=10, initial_states_path="DEFAULT", seed=7,
            pretrained_checkpoint="checkpoint", model_family="openvla", env_img_res=256,
            save_video=False, use_wandb=False,
        )
        suite = SimpleNamespace(n_tasks=2, get_task=lambda i: SimpleNamespace(name=f"task_{i}"),
                                get_task_init_states=lambda i: [i, i])
        self.environments = [Mock(), Mock()]
        self.log_file = io.StringIO()
        self.backend = SimpleNamespace(
            benchmark=SimpleNamespace(get_benchmark_dict=lambda: {self.suite: lambda: suite}),
            validate_config=Mock(), set_seed_everywhere=Mock(), initialize_model=Mock(return_value=(None,) * 5),
            get_image_resize_size=Mock(return_value=224), setup_logging=Mock(return_value=(self.log_file, "", "")),
            get_libero_env=Mock(side_effect=[(env, "task") for env in self.environments]),
            run_episode=Mock(side_effect=lambda *args, **kwargs: (kwargs["initial_state"] == 0, [])),
            log_message=Mock(),
        )
        self.path = reports.task_results_path(self.output, "test", self.suite)

    def evaluate(self):
        return runner.evaluate(self.cfg, self.options, self.metadata, self.backend)

    def records(self):
        return reports.read_task_results(self.path, self.suite, self.metadata[self.suite])

    def test_native_rollout_and_complete_resume_does_not_load_model(self):
        self.assertEqual(self.evaluate(), 0.5)
        self.assertEqual(len(self.records()), 2)
        self.assertTrue(self.log_file.closed)
        for env in self.environments:
            env.close.assert_called_once()
        self.assertTrue(self.backend.run_episode.call_args.kwargs["raise_on_error"])
        self.options.resume = True
        self.backend.initialize_model.reset_mock()
        self.assertEqual(self.evaluate(), 0.5)
        self.backend.initialize_model.assert_not_called()

    def test_interruption_saves_only_complete_tasks_and_closes_environment(self):
        self.backend.run_episode.side_effect = [(True, []), RuntimeError("inference failed")]
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.evaluate()
        self.assertEqual([item["task_name"] for item in self.records()], ["task_0"])
        self.environments[1].close.assert_called_once()
        self.assertTrue(self.log_file.closed)
        self.options.resume = True
        self.backend.get_libero_env.side_effect = [(Mock(), "task")]
        self.backend.run_episode.side_effect = [(False, [])]
        self.assertEqual(self.evaluate(), 0.5)
        self.assertEqual(len(self.records()), 2)

    def test_changed_trial_count_reruns_tasks_and_fresh_run_clears_old_records(self):
        self.evaluate()
        self.options.resume = True
        self.cfg.num_trials_per_task = 2
        self.backend.get_libero_env.side_effect = [(Mock(), "task"), (Mock(), "task")]
        self.backend.run_episode.reset_mock()
        self.assertEqual(self.evaluate(), 0.5)
        self.assertEqual(self.backend.run_episode.call_count, 4)
        self.assertEqual([item["episodes"] for item in self.records()], [2, 2])
        self.options.resume = False
        self.backend.get_libero_env.side_effect = [(Mock(), "task")]
        self.backend.run_episode.side_effect = RuntimeError("failed")
        with self.assertRaises(RuntimeError):
            self.evaluate()
        self.assertEqual(self.records(), [])

    def test_rejects_wrong_benchmark_and_invalid_trials_before_loading_model(self):
        self.metadata[self.suite]["missing"] = {"id": 2, "category": "lighting"}
        with self.assertRaisesRegex(ValueError, "Task/classification mismatch"):
            self.evaluate()
        self.cfg.num_trials_per_task = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            self.evaluate()
        self.backend.initialize_model.assert_not_called()

    def test_initial_state_count_is_checked(self):
        self.cfg.num_trials_per_task = 3
        with self.assertRaisesRegex(ValueError, "only 2 initial states"):
            self.evaluate()
        self.backend.initialize_model.assert_not_called()

    def test_atomic_write_preserves_progress_on_failure(self):
        self.evaluate()
        before = self.path.read_bytes()
        with patch.object(reports.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                reports.write_task_results(self.path, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_category_is_rejected(self):
        self.evaluate()
        records = self.records()
        records[0]["category"] = "invalid"
        reports.write_task_results(self.path, records)
        with self.assertRaisesRegex(ValueError, "Category mismatch"):
            self.records()

    def test_offline_summary_weights_and_incomplete_coverage(self):
        classification = self.output / "classification.json"
        classification.write_text(json.dumps({
            suite: [dict(name=name, **item) for name, item in tasks.items()]
            for suite, tasks in self.metadata.items()
        }), encoding="utf-8")
        for index, suite in enumerate(reports.LIBERO_PLUS_SUITES):
            reports.write_task_results(reports.task_results_path(self.output, "test", suite), [{
                "suite": suite, "task_name": "task_0", "category": "lighting",
                "episodes": 9 if index == 0 else 1, "successes": 9 if index == 0 else 0,
            }])
        result = subprocess.run([
            sys.executable, "-S", str(Path(runner.__file__).resolve()), "--summary_only", "--save_name", "test",
            "--output_dir", str(self.output), "--classification_path", str(classification),
        ], capture_output=True, text=True, cwd=self.output)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = (self.output / "results/test_plus_category_summary.md").read_text(encoding="utf-8")
        self.assertIn("| **All cases** | 4 | 8 | 12 | 9 | 75.0% |", text)
        self.assertIn("| lighting | 25.0% | 100.0% | 0.0% | 0.0% | 0.0% |", text)
        self.assertEqual(text.count("WARNING:"), 4)

    def test_summary_requires_all_suites(self):
        with self.assertRaises(FileNotFoundError):
            reports.write_summary(self.output, "test", self.metadata)

    def test_noninteractive_config_and_missing_checkout(self):
        with self.assertRaises(FileNotFoundError):
            reports.validate_checkout(self.output)
        config = reports.ensure_libero_config(self.output / "checkout", self.output / "config")
        original = config.read_bytes()
        reports.ensure_libero_config(self.output / "other", self.output / "config")
        self.assertEqual(config.read_bytes(), original)

    def test_native_episode_errors_propagate_only_when_requested(self):
        # Execute the actual episode function without importing GPU libraries.
        path = Path(runner.__file__).with_name("run_libero_eval.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_episode")
        namespace = dict(GenerateConfig=object, NUM_ACTIONS_CHUNK=8, deque=deque,
                         TASK_MAX_STEPS={self.suite: 1}, log_message=Mock(),
                         get_libero_dummy_action=lambda _: [0] * 7)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        env = Mock()
        env.step.side_effect = RuntimeError("simulation error")
        success, _ = namespace["run_episode"](self.cfg, env, "task", None, 224)
        self.assertFalse(success)
        with self.assertRaisesRegex(RuntimeError, "simulation error"):
            namespace["run_episode"](self.cfg, env, "task", None, 224, raise_on_error=True)


if __name__ == "__main__":
    unittest.main()
