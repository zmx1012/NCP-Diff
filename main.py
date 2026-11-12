"""Entry point for NCP-Diff experiments on SMD, PSM, MSL and SMAP."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import List

import numpy as np
import pandas as pd
import torch

from data_loader import get_dataloaders
from train_test import evaluate_model, load_model_for_test, read_test_checkpoint


def build_parser():
    p = argparse.ArgumentParser(
        description="Unsupervised multivariate time-series anomaly detection with prototype-guided latent diffusion."
    )
    # Choose independent training and inference workflows.
    p.add_argument("--mode", type=str, default="test", choices=["train", "test", "train_test"],
                   help="train: train only; test: load an existing checkpoint without training; train_test: both.")
    p.add_argument("--checkpoint_path", type=str, default="./outputs_ncp_diff",
                   help="Optional .pt checkpoint for --mode test (only one dataset/seed). "
                        "Otherwise use <output_dir>/<dataset>/seed_<seed>/best_model.pt.")
    # Data. data_root is the absolute parent directory; the loader appends the dataset name.
    p.add_argument("--dataset", type=str, default="ALL", choices=["SMD", "PSM", "MSL", "SMAP", "CUSTOM", "ALL", "all"])
    p.add_argument("--data_root", type=str, default=r"E:/Work/pytorch/Diffusion-Models/mydiffusion/dataset",
                   help=r"Absolute parent path, e.g. D:\datasets; standard data folders are D:\datasets\SMD, D:\datasets\PSM, etc.")
    p.add_argument("--custom_data_dir", type=str, default="",
                   help="Optional absolute folder containing custom train.csv, test.csv, test_label.csv; used with --dataset CUSTOM. If omitted, uses <data_root>/CUSTOM.")
    p.add_argument("--output_dir", type=str, default="./outputs_ncp_diff")
    p.add_argument("--custom_drop_first_column", type=int, choices=[0, 1], default=0,
                   help="Set 1 when CUSTOM train/test CSVs have an unnamed timestamp in column 0. PSM always drops column 0.")
    p.add_argument("--use_random_data", type=int, default=0,
                   help="1=use synthetic random data for smoke tests; 0=load the real dataset files.")
    p.add_argument("--random_train_length", type=int, default=4000)
    p.add_argument("--random_test_length", type=int, default=2000)
    p.add_argument("--random_features", type=int, default=8)
    p.add_argument("--train_fraction", type=float, default=0.90,
                   help="Chronological fraction of the normal training series used for model fitting.")

    # Multi-seed reproducibility.
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    p.add_argument("--deterministic", type=int, default=1)

    # Windowing / dataloaders.
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--train_stride", type=int, default=1)
    p.add_argument("--val_stride", type=int, default=64)
    p.add_argument("--test_stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)

    # Architecture.
    p.add_argument("--d_model", type=int, default=16)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--context_layers", type=int, default=2)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--context_views", type=int, default=2,
                   help="Complementary blind-spot context views; each token masked exactly once.")
    p.add_argument("--context_block", type=int, default=8,
                   help="Contiguous time block size masked in each blind-spot view.")
    p.add_argument("--ema_momentum", type=float, default=0.99)
    p.add_argument("--risk_target_scale", type=float, default=0.5)

    # Ablation switches (0=off, 1=on).
    p.add_argument("--use_context_encoder", type=int, default=1)
    p.add_argument("--use_prototypes", type=int, default=1)
    p.add_argument("--use_contamination_gate", type=int, default=1)
    p.add_argument("--use_diffusion", type=int, default=1)
    p.add_argument("--use_spatiotemporal_attention", type=int, default=1)
    p.add_argument("--use_synthetic_corruption", type=int, default=1)

    # Adaptive normal prototype bank.
    p.add_argument("--n_prototypes", type=int, default=8, help="Initial active prototype count.")
    p.add_argument("--min_prototypes", type=int, default=2)
    p.add_argument("--max_prototypes", type=int, default=32)
    p.add_argument("--prototype_temperature", type=float, default=0.15)
    p.add_argument("--prototype_expand_threshold", type=float, default=0.30)
    p.add_argument("--prototype_prune_threshold", type=float, default=0.002)
    p.add_argument("--prototype_prune_dist", type=float, default=0.12)
    p.add_argument("--prototype_warmup", type=int, default=3)
    p.add_argument("--prototype_update_interval", type=int, default=2)
    p.add_argument("--trust_temperature", type=float, default=0.50)
    p.add_argument("--prototype_patch_len", type=int, default=64,
                   help="Length P for temporal templates [K,P,F]; sensor templates [K,D,F].")
    p.add_argument("--prototype_temporal_weight", type=float, default=0.5,
                   help="Relative matching contribution of temporal template, in [0,1].")
    p.add_argument("--prototype_interaction_weight", type=float, default=0.1,
                   help="Low-rank time x sensor interaction gain, 0 disables interaction.")

    # Latent diffusion.
    p.add_argument("--diffusion_steps", type=int, default=20)
    p.add_argument("--diffusion_start_t", type=int, default=10,
                   help="Partial diffusion start step; also limits training noise levels.")
    p.add_argument("--diffusion_hidden_dim", type=int, default=64,
                   help="Legacy option retained for CLI compatibility; v2 denoiser uses d_model.")
    p.add_argument("--denoiser_layers", type=int, default=1)
    p.add_argument("--diffusion_beta_start", type=float, default=0.0001)
    p.add_argument("--diffusion_beta_end", type=float, default=0.02)

    # Synthetic corruption and objective.
    p.add_argument("--synthetic_corruption_prob", type=float, default=0.70)
    p.add_argument("--max_corrupt_segments", type=int, default=2)
    p.add_argument("--corrupt_region_weight", type=float, default=1.0)
    p.add_argument("--lambda_background", type=float, default=0.10)
    p.add_argument("--lambda_prototype", type=float, default=0.05)
    p.add_argument("--lambda_diffusion", type=float, default=0.10)
    p.add_argument("--lambda_context", type=float, default=0.20)
    p.add_argument("--lambda_latent", type=float, default=0.20)
    p.add_argument("--lambda_nonlocal", type=float, default=0.20)
    p.add_argument("--lambda_risk", type=float, default=0.10)
    p.add_argument("--lambda_correction_magnitude", type=float, default=0.05)
    p.add_argument("--lambda_restored_latent", type=float, default=0.20)
    p.add_argument("--run_propagation_probe", type=int, default=1)
    p.add_argument("--probe_max_batches", type=int, default=3)

    # Anomaly score composition and thresholding.
    p.add_argument("--score_prototype_weight", type=float, default=0.25)
    p.add_argument("--score_contamination_weight", type=float, default=0.10)
    p.add_argument("--score_conditional_weight", type=float, default=0.25)
    p.add_argument("--score_aggregation", type=str, choices=["mean", "max", "topk"], default="topk")
    p.add_argument("--score_topk", type=int, default=3)
    p.add_argument("--threshold_quantile", type=float, default=0.8,
                   help="Threshold quantile estimated only from the normal validation scores.")
    p.add_argument("--compute_vus", type=int, default=1)
    p.add_argument("--save_score_components", type=int, choices=[0, 1], default=1,
                   help="Save all [time,variable] calibration and test score components (may be large).")
    # PaAno's metric formulas are bundled inside metrics.py. No clone needed.
    p.add_argument("--profile_flops", type=int, choices=[0, 1], default=1,
                   help="Run partial PyTorch-profiler FLOPs on one real test batch (1/0).")
    p.add_argument("--paano_vus_thresholds", type=int, default=250,
                   help="Number of PaAno VUS operating thresholds (original default 250).")
    p.add_argument("--vus_window", type=int, default=100,
                   help="Sliding-window parameter for the optional VUS package; fixed before test evaluation.")

    # Optimization.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--gradient_clip", type=float, default=5.0)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--lr_patience", type=int, default=2)
    p.add_argument("--early_stopping", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-6)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    return p


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def setup_logger(log_path: str, name: str = "NCPDiff"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _metric_summary(rows: List[dict], output_dir: str, dataset: str, logger):
    df = pd.DataFrame(rows)
    seed_csv = os.path.join(output_dir, f"{dataset}_seed_results.csv")
    df.to_csv(seed_csv, index=False, encoding="utf-8-sig")
    numeric_cols = [c for c in df.columns if c not in ("dataset", "seed") and pd.api.types.is_numeric_dtype(df[c])]
    summary_rows = []
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            mean, std = float("nan"), float("nan")
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        summary_rows.append({"metric": col, "mean": mean, "std": std, "n_valid": int(vals.size)})
        logger.info("Across seeds %-28s mean=%.6f std=%.6f (n=%d)", col, mean, std, vals.size)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, f"{dataset}_mean_std.csv"), index=False, encoding="utf-8-sig")
    return df, summary_df


# Keys that have to match the training run to reconstruct the same architecture,
# inference behavior, data split, and synthetic smoke-test data.
# Score weights, threshold, batch size, device and logging remain adjustable at test time.
_TRAIN_CONFIG_KEYS = (
    "train_fraction", "window_size", "val_stride", "test_stride",
    "use_random_data", "random_train_length", "random_test_length", "random_features",
    "custom_drop_first_column", "d_model", "n_heads", "encoder_layers",
    "context_layers", "decoder_layers", "dropout", "context_views", "context_block",
    "ema_momentum", "risk_target_scale", "use_context_encoder", "use_prototypes",
    "use_contamination_gate", "use_diffusion", "use_spatiotemporal_attention",
    "use_synthetic_corruption", "n_prototypes", "min_prototypes", "max_prototypes",
    "prototype_temperature", "prototype_expand_threshold", "prototype_prune_threshold",
    "prototype_prune_dist", "prototype_warmup", "prototype_update_interval",
    "trust_temperature", "prototype_patch_len", "prototype_temporal_weight",
    "prototype_interaction_weight", "diffusion_steps", "diffusion_start_t", "denoiser_layers",
    "diffusion_beta_start", "diffusion_beta_end",
)


def _restore_test_config(args, checkpoint, checkpoint_path, dataset, logger):
    """Load the trained architecture rather than silently using CLI defaults.

    New checkpoints embed model_args. Older checkpoints used dataset/config.json.
    Never overwrite the original training configuration when testing.
    """
    saved = checkpoint.get("model_args") if isinstance(checkpoint, dict) else None
    source = "checkpoint:model_args"
    if not isinstance(saved, dict):
        config_paths = [
            os.path.join(os.path.dirname(checkpoint_path), "config.json"),
            os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.json"),
        ]
        saved = None
        for path in config_paths:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                source = path
                break
    if not isinstance(saved, dict):
        raise FileNotFoundError(
            "Missing checkpoint model_args and matching training config.json. "
            "Cannot safely reconstruct the model for test-only mode. "
            "Use a new checkpoint or restore the dataset/config.json saved by training."
        )
    ckpt_dataset = str(saved.get("dataset", dataset)).upper()
    if ckpt_dataset != dataset.upper():
        raise ValueError(f"Checkpoint was trained on {ckpt_dataset}, but --dataset={dataset}.")
    updated = []
    for key in _TRAIN_CONFIG_KEYS:
        if key in saved:
            if getattr(args, key, None) != saved[key]:
                updated.append(key)
            setattr(args, key, saved[key])
    logger.info("Restored model/split configuration from %s (updated %d values: %s)",
                source, len(updated), ", ".join(updated) if updated else "none")
    logger.info("Test-time score settings remain configurable: threshold_quantile=%.4f aggregation=%s topk=%d",
                args.threshold_quantile, args.score_aggregation, args.score_topk)
    return saved


def _validate_args(args, parser):
    if args.d_model % args.n_heads != 0:
        parser.error("--d_model must be divisible by --n_heads")
    if not (0.5 < args.train_fraction < 1.0):
        parser.error("--train_fraction must be in (0.5, 1.0)")
    if not (0.0 < args.threshold_quantile < 1.0):
        parser.error("--threshold_quantile must be in (0, 1)")
    if args.prototype_patch_len < 1 or args.prototype_patch_len > args.window_size:
        parser.error("--prototype_patch_len must be in [1, window_size]")
    if not (0.0 <= args.prototype_temporal_weight <= 1.0):
        parser.error("--prototype_temporal_weight must be in [0,1]")
    if args.paano_vus_thresholds < 2:
        parser.error("--paano_vus_thresholds must be >= 2")
    if args.max_prototypes < args.n_prototypes:
        parser.error("--max_prototypes must be >= --n_prototypes")
    if args.context_views < 2 or args.context_block < 1:
        parser.error("--context_views >= 2 and --context_block >= 1 are required")
    if not (0.0 <= args.ema_momentum < 1.0):
        parser.error("--ema_momentum must be in [0,1)")
    if min(args.train_stride, args.val_stride, args.test_stride) < 1:
        parser.error("All strides must be positive")
    if max(args.train_stride, args.val_stride, args.test_stride) > args.window_size:
        parser.error("All strides must be <= window_size to cover every timestamp")


def main():
    parser = build_parser()
    args = parser.parse_args()
    # Test-only should only run normal-score calibration and test inference by
    # default. The optional synthetic mechanism probe must be explicitly asked for.
    if args.mode == "test" and not any(a == "--run_propagation_probe" or a.startswith("--run_propagation_probe=") for a in sys.argv[1:]):
        args.run_propagation_probe = 0
    requested_dataset = args.dataset.upper()
    datasets = (["SMAP"] if requested_dataset == "ALL" else [requested_dataset])
    # if args.checkpoint_path and args.mode != "test":
    #     parser.error("--checkpoint_path is only used with --mode test")
    # if args.checkpoint_path and (len(datasets) != 1 or len(args.seeds) != 1):
    #     parser.error("--checkpoint_path requires exactly one dataset and one seed")
    if args.mode != "test":
        _validate_args(args, parser)

    all_dataset_summaries = []
    for dataset in datasets:
        args.dataset = dataset
        dataset_out = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_out, exist_ok=True)
        logger = setup_logger(os.path.join(dataset_out, "experiment.log"), name=f"NCPDiff-{dataset}")
        logger.info("=" * 90)
        logger.info("Mode=%s | Dataset=%s | data_root=%s | output=%s", args.mode, dataset, args.data_root, dataset_out)
        logger.info("Seeds=%s | device=%s | random_data=%d", args.seeds, args.device, args.use_random_data)

        # Save only during actual training; testing must never destroy this file.
        if args.mode in ("train", "train_test"):
            with open(os.path.join(dataset_out, "config.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=2)

        seed_rows = []
        for seed in args.seeds:
            args.seed = int(seed)
            set_seed(int(seed), bool(args.deterministic))
            run_dir = os.path.join(dataset_out, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            checkpoint = None
            checkpoint_path = ""
            if args.mode == "test":
                checkpoint_path = (args.checkpoint_path if args.checkpoint_path.lower().endswith(".pt")
                                   else os.path.join(args.checkpoint_path or args.output_dir,
                                                     dataset, f"seed_{seed}", "checkpoint_epoch_001.pt"))
                # Read weights + saved training configuration before creating model or data loaders.
                checkpoint = read_test_checkpoint(checkpoint_path)
                _restore_test_config(args, checkpoint, checkpoint_path, dataset, logger)
                _validate_args(args, parser)
                # Keep reproducibility settings for this test separate from training config.
                with open(os.path.join(run_dir, "test_config.json"), "w", encoding="utf-8") as f:
                    json.dump({"checkpoint_path": checkpoint_path, **vars(args)}, f,
                              ensure_ascii=False, indent=2)

            logger.info("\n--- Seed %d: %s data loading ---", seed, args.mode)
            loaders, meta = get_dataloaders(args, logger)
            logger.info("Feature dimension=%d | train/val/test lengths=%d/%d/%d",
                        args.input_dim, meta["train_length"], meta["val_length"], meta["test_length"])

            if args.mode == "test":
                model, train_info = load_model_for_test(args, checkpoint, checkpoint_path, logger)
            else:
                # Import training only in a mode that actually trains.
                from train_test import train_model
                model, train_info = train_model(args, loaders, logger, run_dir)

            row = {
                "dataset": dataset,
                "seed": int(seed),
                "best_epoch": int(train_info["best_epoch"]),
                "best_val_loss": float(train_info["best_val_loss"]),
                "active_prototypes": int(model.proto.num_active) if model.proto is not None else 0,
            }
            if args.mode in ("test", "train_test"):
                # Test-only DOES NOT call train_model, run optimizer, or train for one step.
                test_metrics = evaluate_model(args, model, loaders, meta, run_dir, logger)
                row.update(test_metrics)
                logger.info("Seed %d tested. PA_F1=%.4f Raw_F1=%.4f", seed,
                            row.get("PA_F1", float("nan")), row.get("Raw_F1", float("nan")))
            else:
                logger.info("Seed %d trained only; best_epoch=%d val_loss=%.6f checkpoint=%s", seed,
                            train_info["best_epoch"], train_info["best_val_loss"],
                            os.path.join(run_dir, "best_model.pt"))
            seed_rows.append(row)

        _, summary_df = _metric_summary(seed_rows, dataset_out, dataset, logger)
        summary_df.insert(0, "dataset", dataset)
        all_dataset_summaries.append(summary_df)
        logger.info("Finished dataset %s mode=%s", dataset, args.mode)

    if all_dataset_summaries:
        combined = pd.concat(all_dataset_summaries, ignore_index=True)
        os.makedirs(args.output_dir, exist_ok=True)
        combined.to_csv(os.path.join(args.output_dir, "all_datasets_mean_std.csv"),
                        index=False, encoding="utf-8-sig")
        print(f"\nFinished {args.mode}. Results saved under: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
