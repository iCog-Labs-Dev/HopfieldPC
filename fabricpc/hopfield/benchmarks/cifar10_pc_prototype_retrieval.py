"""CIFAR-10 identity-versus-Hopfield benchmark using a FabricPC PC graph.

Mirrors ``mnist_pc_prototype_retrieval`` on CIFAR-10: same splits, prototype
construction, beta selection and corruptions, with a small CNN built from
FabricPC nodes and trained with predictive coding. Evaluation reuses the
trained parameters for the identity and frozen-Hopfield bridge graphs.

If the download stalls, place a verified ``cifar-10-python.tar.gz`` in ``data/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import KaimingInitializer, XavierInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.hopfield.benchmarks import mnist_baseline as mb
from fabricpc.hopfield.benchmarks.mnist_pc_prototype_retrieval import (
    BRIDGE_NODE,
    DEFAULT_ETA_INFER,
    DEFAULT_INFER_STEPS,
    EMBEDDING_NODE,
    INPUT_NODE,
    OUTPUT_NODE,
    _assert_same_parameter_layout,
    graph_outputs,
)
from fabricpc.hopfield.benchmarks.mnist_prototype_retrieval import build_memory
from fabricpc.nodes import ConvNode, HopfieldMemoryNode, IdentityNode, Linear, MaxPool
from fabricpc.training import train

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "cifar10_prototype_retrieval.yaml"
)


def _md5(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "md5").hexdigest()


def _cifar10_archive(data_dir: Path) -> Path:
    archive = data_dir / "cifar-10-python.tar.gz"
    if archive.is_file() and _md5(archive) == CIFAR10_MD5:
        return archive
    data_dir.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".part")
    print(f"Downloading CIFAR-10 (~170 MB) to {archive} ...", flush=True)
    urllib.request.urlretrieve(CIFAR10_URL, partial)
    if _md5(partial) != CIFAR10_MD5:
        partial.unlink()
        raise RuntimeError("Downloaded CIFAR-10 archive failed its MD5 check.")
    return partial.replace(archive)


def load_cifar10(data_dir: str | Path) -> tuple[jax.Array, ...]:
    """Return train/test images (NCHW float32 in [0, 1]) and int32 labels."""
    batches = {}
    with tarfile.open(_cifar10_archive(Path(data_dir))) as tar:
        for member in tar:
            name = Path(member.name).name
            if name.startswith("data_batch_") or name == "test_batch":
                batches[name] = pickle.load(tar.extractfile(member), encoding="bytes")

    def stack(names):
        x = np.concatenate([batches[n][b"data"] for n in names]).reshape(-1, 3, 32, 32)
        y = np.concatenate([batches[n][b"labels"] for n in names])
        return jnp.asarray(x, dtype=jnp.float32) / 255.0, jnp.asarray(y, jnp.int32)

    train_x, train_y = stack([f"data_batch_{i}" for i in range(1, 6)])
    test_x, test_y = stack(["test_batch"])
    return train_x, train_y, test_x, test_y


def build_pc_cifar10_graph(
    *,
    prototypes: jax.Array | None = None,
    center: jax.Array | None = None,
    beta: float = 1.0,
    num_updates: int = 1,
    infer_steps: int = DEFAULT_INFER_STEPS,
    eta_infer: float = DEFAULT_ETA_INFER,
    c1: int = 32,
    c2: int = 64,
    hidden: int = 128,
    embed_dim: int = 64,
):
    """Same CNN as the MNIST benchmark on 32x32x3 inputs (32 -> 16 -> 8)."""
    relu_init = KaimingInitializer(nonlinearity="relu")
    pixels = IdentityNode(shape=(32, 32, 3), name=INPUT_NODE)
    conv1 = ConvNode(
        shape=(32, 32, c1),
        kernel_size=(3, 3),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=relu_init,
        name="conv1",
    )
    pool1 = MaxPool(
        shape=(16, 16, c1),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool1",
    )
    conv2 = ConvNode(
        shape=(16, 16, c2),
        kernel_size=(3, 3),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=relu_init,
        name="conv2",
    )
    pool2 = MaxPool(
        shape=(8, 8, c2),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool2",
    )
    dense = Linear(
        shape=(hidden,),
        activation=ReLUActivation(),
        flatten_input=True,
        weight_init=relu_init,
        name="dense",
    )
    embedding = Linear(
        shape=(embed_dim,), weight_init=XavierInitializer(), name=EMBEDDING_NODE
    )
    if prototypes is None:
        bridge = IdentityNode(shape=(embed_dim,), name=BRIDGE_NODE)
    else:
        bridge = HopfieldMemoryNode(
            shape=(embed_dim,),
            name=BRIDGE_NODE,
            prototypes=prototypes,
            beta=beta,
            num_updates=num_updates,
            normalize=True,
            center=center,
        )
    classifier = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer(),
        name=OUTPUT_NODE,
    )
    chain = [pixels, conv1, pool1, conv2, pool2, dense, embedding, bridge, classifier]
    return graph(
        nodes=chain,
        edges=[Edge(source=a, target=b.slot("in")) for a, b in zip(chain, chain[1:])],
        task_map=TaskMap(x=pixels, y=classifier),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )


def prepare_images(images: jax.Array) -> jax.Array:
    """Convert CIFAR-10 NCHW pixels to normalized FabricPC NHWC input."""
    images = jnp.asarray(images, dtype=jnp.float32)
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError(
            f"CIFAR-10 images must have shape [N, 3, 32, 32], got {tuple(images.shape)}."
        )
    mean = jnp.asarray(CIFAR10_MEAN).reshape(1, 1, 1, 3)
    std = jnp.asarray(CIFAR10_STD).reshape(1, 1, 1, 3)
    return (jnp.transpose(images, (0, 2, 3, 1)) - mean) / std


class Cifar10Loader:
    """Re-iterable seeded minibatches for FabricPC's training API."""

    def __init__(
        self,
        images: jax.Array,
        labels: jax.Array,
        *,
        batch_size: int,
        seed: int,
        device: jax.Device,
    ) -> None:
        self.images = prepare_images(images)
        self.labels = jnp.asarray(labels, dtype=jnp.int32)
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(seed)
        self.device = device

    def __len__(self) -> int:
        return math.ceil(self.images.shape[0] / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        self.key, permutation_key = jax.random.split(self.key)
        permutation = jax.random.permutation(permutation_key, self.images.shape[0])
        for start in range(0, self.images.shape[0], self.batch_size):
            indices = permutation[start : start + self.batch_size]
            yield {
                "x": jax.device_put(self.images[indices], self.device),
                "y": jax.device_put(self.labels[indices], self.device),
            }


def make_eval_batch(graph_structure):
    @jax.jit
    def evaluate_batch(params, images, key):
        # Feed-forward evaluation, as in the MNIST benchmark: PC is used for
        # training only.
        x = prepare_images(images)
        state = initialize_graph_state(
            graph_structure,
            batch_size=x.shape[0],
            rng_key=key,
            clamps={INPUT_NODE: x},
            params=params,
        )
        return (
            jnp.argmax(state.nodes[OUTPUT_NODE].z_mu, axis=-1),
            state.nodes[EMBEDDING_NODE].z_latent,
            state.nodes[BRIDGE_NODE].z_mu,
        )

    return evaluate_batch


def _mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def write_reports(payload: dict[str, Any], report_dir: Path) -> None:
    """Write results.json, results.csv and summary.md for one run or merged runs."""
    cfg, rows, seeds = payload["config"], payload["rows"], payload["seeds"]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    fieldnames = [
        "seed",
        "experiment",
        "variant",
        "kind",
        "level",
        "head_acc",
        "proto_acc",
        "dist_to_true_proto",
        "entropy_norm",
        "top1_weight",
        "true_proto_weight",
    ]
    with open(report_dir / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    cells: dict[tuple, dict[int, dict[str, dict]]] = {}
    for row in rows:
        key = (row["experiment"], row["kind"], row["level"])
        cells.setdefault(key, {}).setdefault(row["seed"], {})[row["variant"]] = row

    summary = [
        "# FabricPC PC CIFAR-10 Baseline vs Hopfield",
        "",
        f"- Seeds: `{[row['seed'] for row in seeds]}`, epochs: `{cfg['epochs']}`, "
        f"batch size: `{cfg['batch_size']}`, lr: `{cfg['lr']}`",
        f"- PC inference: `{cfg['pc_infer_steps']}` steps at eta `{cfg['pc_eta_infer']}`",
        f"- Prototype memory: `{cfg['prototypes_per_class']} per class`, "
        f"centered `{cfg['center_prototypes']}`",
        "- Test head accuracy, mean ± std over seeds; Δ is the paired per-seed "
        "difference (Hopfield − Baseline).",
        "",
        "| Experiment | Corruption | Level | Baseline | Hopfield | Δ | Seeds Hopfield wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (experiment, kind, level), by_seed in cells.items():
        if kind != "clean" and not level:
            continue  # level 0 repeats the clean result
        base = [v["baseline"]["head_acc"] for v in by_seed.values()]
        hop = [v["hopfield"]["head_acc"] for v in by_seed.values()]
        delta = [h - b for b, h in zip(base, hop)]
        (b_mean, b_std), (h_mean, h_std) = _mean_std(base), _mean_std(hop)
        d_mean, d_std = _mean_std(delta)
        summary.append(
            f"| {experiment} | {kind} | {level:g} | {b_mean:.4f} ± {b_std:.4f} "
            f"| {h_mean:.4f} ± {h_std:.4f} | {d_mean:+.4f} ± {d_std:.4f} "
            f"| {sum(d > 0 for d in delta)}/{len(delta)} |"
        )

    clean = cells[("E1_clean", "clean", 0.0)]
    summary += ["", "Hopfield diagnostics, clean test set (mean ± std):", ""]
    for metric in ("proto_acc", "entropy_norm", "top1_weight", "true_proto_weight"):
        mean, std = _mean_std([v["hopfield"][metric] for v in clean.values()])
        summary.append(f"- {metric}: {mean:.4f} ± {std:.4f}")

    summary += [
        "",
        "| Seed | Beta* | Train time (s) | Clean baseline | Clean Hopfield |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in seeds:
        summary.append(
            f"| {row['seed']} | {row['beta_star']:g} | {row['train_time_s']:.1f} "
            f"| {row['E1_clean_baseline']:.4f} | {row['E1_clean_hopfield']:.4f} |"
        )
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def run_benchmark(cfg: dict[str, Any], *, infer_steps: int, eta_infer: float):
    device = mb.resolve_device(cfg["device"])
    mb.configure_threads(cfg["cost"]["num_threads"])
    print(
        f"Loading CIFAR-10 from {cfg['data_dir']} (downloading if needed)...",
        flush=True,
    )
    data = load_cifar10(cfg["data_dir"])
    print(f"CIFAR-10 loaded; running on {device}.", flush=True)
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    grids = mb.noise_grids(cfg)
    experiments = [("E1_clean", "clean", 0.0)]
    for tag, kind in (
        ("E3a", "gaussian"),
        ("E3b", "salt_pepper"),
        ("E3c", "occlusion"),
    ):
        experiments += [(f"{tag}_{lv:g}", kind, lv) for lv in grids[kind] if lv]

    for seed in cfg["seeds"]:
        splits = mb.make_splits(
            data,
            seed=seed,
            val_size=cfg["val_size"],
            train_subset=cfg["train_subset"],
            test_subset=cfg["test_subset"],
        )
        structure = build_pc_cifar10_graph(infer_steps=infer_steps, eta_infer=eta_infer)
        loader = Cifar10Loader(
            splits.train_images,
            splits.train_labels,
            batch_size=cfg["batch_size"],
            seed=seed,
            device=device,
        )
        print(
            f"Seed {seed}: training with PC on {len(splits.train_images)} images.",
            flush=True,
        )
        started = time.perf_counter()
        trained = train(
            params=initialize_params(structure, jax.random.PRNGKey(seed)),
            structure=structure,
            train_loader=loader,
            optimizer=optax.adam(cfg["lr"]),
            config={"num_epochs": cfg["epochs"]},
            rng_key=jax.random.PRNGKey(seed + 1),
            algorithm="pc",
            verbose=True,
        )
        params, train_time = trained.params, time.perf_counter() - started
        del loader
        print(f"Seed {seed}: PC training finished in {train_time:.1f}s.", flush=True)

        baseline_eval = make_eval_batch(structure)
        step = cfg["eval_batch_size"]
        embeddings = [
            baseline_eval(
                params,
                jax.device_put(splits.train_images[start : start + step], device),
                jax.random.PRNGKey(seed + start),
            )[1]
            for start in range(0, len(splits.train_images), step)
        ]
        memory, memory_labels, center = build_memory(
            jnp.concatenate(embeddings, axis=0),
            splits.train_labels,
            per_class=cfg["prototypes_per_class"],
            center=cfg["center_prototypes"],
            seed=seed,
        )

        def hopfield_graph(beta: float):
            return build_pc_cifar10_graph(
                prototypes=memory,
                center=center,
                beta=beta,
                infer_steps=infer_steps,
                eta_infer=eta_infer,
            )

        def corrupted(images, kind, level):
            if kind == "clean" or level == 0:
                return images
            return mb.make_corrupted_test_set(images, seed=seed, kind=kind, level=level)

        def evaluate(graph_structure, eval_fn, images, labels, beta):
            outputs = graph_outputs(
                graph_structure,
                params,
                eval_fn,
                images,
                batch_size=step,
                device=device,
                prototypes=memory,
                memory_labels=memory_labels,
                center=center,
                beta=beta,
                num_updates=1,
            )
            return mb.summarize_outputs(outputs, labels, memory, memory_labels)[0]

        _assert_same_parameter_layout(
            params,
            initialize_params(
                hopfield_graph(cfg["beta_values"][0]), jax.random.PRNGKey(seed)
            ),
        )
        best_beta, best_score = cfg["beta_values"][0], -float("inf")
        for index, beta in enumerate(cfg["beta_values"], start=1):
            print(
                f"Seed {seed}: selecting beta ({index}/{len(cfg['beta_values'])}).",
                flush=True,
            )
            candidate = hopfield_graph(beta)
            candidate_eval = make_eval_batch(candidate)
            score = float(
                np.mean(
                    [
                        evaluate(
                            candidate,
                            candidate_eval,
                            corrupted(splits.val_images, kind, level),
                            splits.val_labels,
                            beta,
                        )["head_acc"]
                        for kind, level in cfg["beta_selection_sets"]
                    ]
                )
            )
            if score > best_score + 1e-12:
                best_beta, best_score = beta, score
        print(
            f"Seed {seed}: selected beta={best_beta:g} (val acc {best_score:.4f}).",
            flush=True,
        )

        hopfield = hopfield_graph(best_beta)
        variants = (
            ("baseline", structure, baseline_eval),
            ("hopfield", hopfield, make_eval_batch(hopfield)),
        )
        seed_row = {"seed": seed, "beta_star": best_beta, "train_time_s": train_time}
        for number, (experiment, kind, level) in enumerate(experiments, start=1):
            print(
                f"Seed {seed}: evaluating {number}/{len(experiments)} {experiment}.",
                flush=True,
            )
            images = corrupted(splits.test_images, kind, level)
            for variant, graph_structure, eval_fn in variants:
                metrics = evaluate(
                    graph_structure, eval_fn, images, splits.test_labels, best_beta
                )
                seed_row[f"{experiment}_{variant}"] = metrics["head_acc"]
                rows.append(
                    {
                        "seed": seed,
                        "experiment": experiment,
                        "variant": variant,
                        "kind": kind,
                        "level": level,
                        **metrics,
                    }
                )
        seed_rows.append(seed_row)

    payload = {
        "config": {
            **cfg,
            "algorithm": "pc",
            "pc_infer_steps": infer_steps,
            "pc_eta_infer": eta_infer,
            "dataset": "CIFAR-10",
        },
        "environment": mb.environment_info(device),
        "seeds": seed_rows,
        "rows": rows,
    }
    write_reports(payload, Path(cfg["reports_dir"]))
    print(f"CIFAR-10 reports saved to: {cfg['reports_dir']}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 identity-versus-Hopfield comparison trained with FabricPC PC."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--infer-steps", type=int, default=DEFAULT_INFER_STEPS)
    parser.add_argument("--eta-infer", type=float, default=DEFAULT_ETA_INFER)
    parser.add_argument("--seeds", type=int, nargs="+", help="override config seeds")
    parser.add_argument("--epochs", type=int, help="override config epochs")
    args = parser.parse_args(argv)
    if args.infer_steps <= 0:
        parser.error("--infer-steps must be positive.")
    if not math.isfinite(args.eta_infer) or args.eta_infer <= 0:
        parser.error("--eta-infer must be positive and finite.")
    if args.seeds and (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0):
        parser.error("--seeds must be unique non-negative integers.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1.")

    cfg = mb.load_config(args.config, quick=args.quick)
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.reports_dir is not None:
        cfg["reports_dir"] = str(args.reports_dir)
    run_benchmark(cfg, infer_steps=args.infer_steps, eta_infer=args.eta_infer)


if __name__ == "__main__":
    main()
