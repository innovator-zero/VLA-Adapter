"""LIBERO-plus evaluation using VLA-Adapter's native policy and rollout code."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robot.libero import libero_plus_utils as reports


def evaluate(cfg, options, metadata, backend):
    """Save only completed tasks; initialize the model only if work remains."""
    suite = cfg.task_suite_name
    if cfg.num_trials_per_task <= 0 or cfg.num_open_loop_steps <= 0 or cfg.num_steps_wait < 0:
        raise ValueError("Trials/open-loop steps must be positive and wait steps nonnegative")
    if cfg.initial_states_path != "DEFAULT":
        raise ValueError("LIBERO-plus evaluation requires the benchmark's DEFAULT initial states")
    backend.validate_config(cfg)
    backend.set_seed_everywhere(cfg.seed)
    task_suite = backend.benchmark.get_benchmark_dict()[suite]()
    reports.validate_task_suite(suite, task_suite, metadata[suite])
    path = reports.task_results_path(options.output_dir, options.save_name, suite)
    records = []
    if options.resume and path.exists():
        for record in reports.read_task_results(path, suite, metadata[suite]):
            if record["episodes"] == cfg.num_trials_per_task:
                records.append(record)
            else:
                logging.warning("Rerunning %s: episode count changed", record["task_name"])
    completed = {record["task_name"] for record in records}
    logging.info("%s: skipping %d/%d completed tasks", suite, len(completed), task_suite.n_tasks)
    reports.write_task_results(path, records)
    components = None
    log_file = None
    try:
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.name in completed:
                continue
            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) < cfg.num_trials_per_task:
                raise ValueError(f"{task.name} has only {len(initial_states)} initial states")
            if components is None:
                if not cfg.pretrained_checkpoint:
                    raise ValueError("Provide --pretrained_checkpoint for evaluation")
                components = backend.initialize_model(cfg)
                resize_size = backend.get_image_resize_size(cfg)
                cfg.local_log_dir = str(options.output_dir / "logs" / options.save_name / suite)
                log_file, _, _ = backend.setup_logging(cfg)
            model, action_head, proprio_projector, noisy_action_projector, processor = components
            env, description = backend.get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            successes = 0
            try:
                for episode in range(cfg.num_trials_per_task):
                    success, images = backend.run_episode(
                        cfg, env, description, model, resize_size,
                        processor=processor, action_head=action_head,
                        proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                        initial_state=initial_states[episode], log_file=log_file, raise_on_error=True,
                    )
                    successes += int(success)
                    backend.log_message(
                        f"Task {task_id} ({task.name}), episode {episode}: success={success}", log_file
                    )
                    if cfg.save_video and images:
                        import imageio

                        video_dir = options.output_dir / "videos" / options.save_name / suite
                        video_dir.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(
                            str(video_dir / f"task_{task_id}_episode_{episode}_success_{int(success)}.mp4"),
                            images, fps=30,
                        )
            finally:
                env.close()
            item = metadata[suite][task.name]
            records.append({
                "suite": suite, "task_id": task_id, "classification_id": item["id"],
                "task_name": task.name, "category": item["category"],
                "episodes": cfg.num_trials_per_task, "successes": successes,
                "success_rate": successes / cfg.num_trials_per_task,
            })
            reports.write_task_results(path, records)
            if cfg.use_wandb:
                backend.wandb.log({f"success_rate/{task.name}": successes / cfg.num_trials_per_task})
            logging.info("%s: completed %d/%d tasks", suite, len(records), task_suite.n_tasks)
        reports.write_suite_summary(options.output_dir, options.save_name, suite, records, metadata[suite])
        rate = reports.success_rate(reports.total_stats(reports.aggregate_records(records).get(suite, {})))
        logging.info("%s success rate: %.4f", suite, rate)
        if components is not None and cfg.use_wandb:
            backend.wandb.log({"success_rate/total": rate})
        return rate
    finally:
        if log_file is not None:
            log_file.close()
        if components is not None and cfg.use_wandb:
            backend.wandb.finish()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--task_suite_name", choices=reports.LIBERO_PLUS_SUITES, default="libero_spatial")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "experiments/logs/libero_plus")
    parser.add_argument("--libero_plus_root", type=Path, default=Path(
        os.environ.get("LIBERO_PLUS_ROOT", str(REPO_ROOT / "third_party/LIBERO-plus"))
    ))
    parser.add_argument("--libero_config_path", type=Path)
    parser.add_argument("--classification_path", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    parser.epilog = "Other options are forwarded to run_libero_eval.GenerateConfig, e.g. --pretrained_checkpoint PATH."
    return parser


def main(argv=None):
    parser = make_parser()
    options, model_args = parser.parse_known_args(argv)
    if not options.save_name or options.save_name in (".", "..") or any(c in options.save_name for c in "/\\"):
        parser.error("--save_name must be a nonempty filename without directory separators")
    root = options.libero_plus_root.expanduser().resolve()
    metadata = reports.load_metadata(options.classification_path or root / reports.CLASSIFICATION_FILE)
    if options.summary_only:
        if model_args:
            parser.error("Summary mode does not accept model options: " + " ".join(model_args))
        reports.write_summary(options.output_dir, options.save_name, metadata)
        return

    reports.validate_checkout(root)
    config_dir = (options.libero_config_path or options.output_dir / ".libero-plus-config").expanduser().resolve()
    reports.ensure_libero_config(root, config_dir)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(root))
    # Select the checkout/config before any LIBERO or model imports.
    from libero.libero import get_libero_path

    for key, relative in (("bddl_files", "bddl_files"), ("init_states", "init_files"), ("assets", "assets")):
        expected = (root / "libero/libero" / relative).resolve()
        if Path(get_libero_path(key)).expanduser().resolve() != expected:
            raise ValueError(f"LIBERO config selects the wrong {key}; use --libero_config_path with a fresh directory")

    import draccus
    from experiments.robot.libero import run_libero_eval as backend

    # Retain all existing model options while using one trial by default for Plus.
    from dataclasses import dataclass

    @dataclass
    class PlusConfig(backend.GenerateConfig):
        num_trials_per_task: int = 1

    cfg = draccus.parse(PlusConfig, args=model_args)
    cfg.task_suite_name = options.task_suite_name
    evaluate(cfg, options, metadata, backend)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted; completed tasks are saved. Use --resume to continue.")
        sys.exit(130)
