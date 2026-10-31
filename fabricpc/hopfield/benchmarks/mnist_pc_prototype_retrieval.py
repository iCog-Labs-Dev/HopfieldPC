"""MNIST identity-versus-Hopfield benchmark using a FabricPC PC graph.

The training protocol follows ``mnist_prototype_retrieval`` for data splits,
architecture widths, prototype construction, beta selection, and corruptions.
The baseline CNN is represented by FabricPC nodes and trained with predictive
coding. Evaluation reuses those trained parameters for both the identity and
frozen-Hopfield bridge graphs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
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
from fabricpc.hopfield import multi_step_retrieve
from fabricpc.hopfield.benchmarks import mnist_baseline as mb
from fabricpc.hopfield.benchmarks.mnist_prototype_retrieval import build_memory
from fabricpc.nodes import (
    ConvNode,
    HopfieldMemoryNode,
    IdentityNode,
    Linear,
    MaxPool,
)
from fabricpc.training import train

INPUT_NODE = "pixels"
EMBEDDING_NODE = "embedding"
BRIDGE_NODE = "bridge"
OUTPUT_NODE = "class"
DEFAULT_INFER_STEPS = 10
DEFAULT_ETA_INFER = 0.05


def build_pc_mnist_graph(
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
    """Build the Task 13 CNN using FabricPC nodes.

    Smaller widths are accepted for unit tests; the benchmark defaults match
    the existing MNIST model's ``[32, 64, 128, 64]`` architecture.
    """
    pixels = IdentityNode(shape=(28, 28, 1), name=INPUT_NODE)
    conv1 = ConvNode(
        shape=(28, 28, c1),
        kernel_size=(3, 3),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=KaimingInitializer(nonlinearity="relu"),
        name="conv1",
    )
    pool1 = MaxPool(
        shape=(14, 14, c1),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool1",
    )
    conv2 = ConvNode(
        shape=(14, 14, c2),
        kernel_size=(3, 3),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=KaimingInitializer(nonlinearity="relu"),
        name="conv2",
    )
    pool2 = MaxPool(
        shape=(7, 7, c2),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool2",
    )
    dense = Linear(
        shape=(hidden,),
        activation=ReLUActivation(),
        flatten_input=True,
        weight_init=KaimingInitializer(nonlinearity="relu"),
        name="dense",
    )
    embedding = Linear(
        shape=(embed_dim,),
        weight_init=XavierInitializer(),
        name=EMBEDDING_NODE,
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

    nodes = [
        pixels,
        conv1,
        pool1,
        conv2,
        pool2,
        dense,
        embedding,
        bridge,
        classifier,
    ]
    edges = [
        Edge(source=pixels, target=conv1.slot("in")),
        Edge(source=conv1, target=pool1.slot("in")),
        Edge(source=pool1, target=conv2.slot("in")),
        Edge(source=conv2, target=pool2.slot("in")),
        Edge(source=pool2, target=dense.slot("in")),
        Edge(source=dense, target=embedding.slot("in")),
        Edge(source=embedding, target=bridge.slot("in")),
        Edge(source=bridge, target=classifier.slot("in")),
    ]
    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=pixels, y=classifier),
        inference=InferenceSGD(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
        ),
    )


def prepare_images(images: jax.Array) -> jax.Array:
    """Convert MNIST NCHW pixels to normalized FabricPC NHWC input."""
    images = jnp.asarray(images, dtype=jnp.float32)
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError(
            "MNIST images must have shape [N, 1, 28, 28], "
            f"got {tuple(images.shape)}."
        )
    images = jnp.transpose(images, (0, 2, 3, 1))
    return (images - mb.MNIST_MEAN) / mb.MNIST_STD


class PcMnistLoader:
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


def _make_eval_batch(graph_structure):
    @jax.jit
    def evaluate_batch(params, images, key):
        # Match the source benchmark's feed-forward evaluation path; PC is used
        # to train the graph, while test-time predictions remain one forward pass.
        x = prepare_images(images)
        clamps = {INPUT_NODE: x}
        state = initialize_graph_state(
            graph_structure,
            batch_size=x.shape[0],
            rng_key=key,
            clamps=clamps,
            params=params,
        )
        return (
            jnp.argmax(state.nodes[OUTPUT_NODE].z_mu, axis=-1),
            state.nodes[EMBEDDING_NODE].z_latent,
            state.nodes[BRIDGE_NODE].z_mu,
        )

    return evaluate_batch


def graph_outputs(
    graph_structure,
    params,
    eval_batch_fn,
    images: jax.Array,
    *,
    batch_size: int,
    device: jax.Device,
    prototypes: jax.Array,
    memory_labels: jax.Array,
    center: jax.Array | None,
    beta: float,
    num_updates: int,
) -> dict[str, jax.Array]:
    """Evaluate one graph in batches and produce the benchmark output schema."""
    outputs = []
    for start in range(0, images.shape[0], batch_size):
        image_batch = jax.device_put(images[start : start + batch_size], device)
        key = jax.random.PRNGKey(start)
        outputs.append(eval_batch_fn(params, image_batch, key))

    head_pred = jnp.concatenate([result[0] for result in outputs], axis=0)
    z = jnp.concatenate([result[1] for result in outputs], axis=0)
    h = jnp.concatenate([result[2] for result in outputs], axis=0)
    result = {"head_pred": head_pred, "z": z, "h": h}
    if isinstance(graph_structure.nodes[BRIDGE_NODE], HopfieldMemoryNode):
        retrieval_query = z if center is None else z - center
        retrieval_memory = prototypes if center is None else prototypes - center
        _, weights = multi_step_retrieve(
            retrieval_query,
            retrieval_memory,
            beta=beta,
            num_updates=num_updates,
            normalize=True,
        )
        num_classes = int(jnp.max(memory_labels)) + 1
        class_mass = jnp.zeros((weights.shape[0], num_classes), dtype=weights.dtype)
        for class_id in range(num_classes):
            class_mass = class_mass.at[:, class_id].set(
                jnp.sum(weights[:, memory_labels == class_id], axis=-1)
            )
        result["weights"] = weights
        result["class_mass"] = class_mass
        result["proto_pred"] = jnp.argmax(class_mass, axis=-1)
    return result


def _assert_same_parameter_layout(left, right) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if jax.tree_util.tree_structure(left) != jax.tree_util.tree_structure(right):
        raise ValueError("Identity and Hopfield graphs have different parameter trees.")
    if [x.shape for x in left_leaves] != [x.shape for x in right_leaves]:
        raise ValueError(
            "Identity and Hopfield graphs have different parameter shapes."
        )


def _run_task(cfg: dict[str, Any], *, infer_steps: int, eta_infer: float):
    device = mb.resolve_device(cfg["device"])
    mb.configure_threads(cfg["cost"]["num_threads"])
    print(
        f"Loading MNIST from {cfg['data_dir']} (downloading missing files if needed)...",
        flush=True,
    )
    data = mb.load_mnist(cfg["data_dir"])
    print(f"MNIST loaded on {device}.", flush=True)
    report_dir = Path(cfg["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed in cfg["seeds"]:
        splits = mb.make_splits(
            data,
            seed=seed,
            val_size=cfg["val_size"],
            train_subset=cfg["train_subset"],
            test_subset=cfg["test_subset"],
        )
        structure = build_pc_mnist_graph(
            infer_steps=infer_steps,
            eta_infer=eta_infer,
        )
        params = initialize_params(structure, jax.random.PRNGKey(seed))
        loader = PcMnistLoader(
            splits.train_images,
            splits.train_labels,
            batch_size=cfg["batch_size"],
            seed=seed,
            device=device,
        )
        optimizer = optax.adam(cfg["lr"])
        print(
            f"Seed {seed}: training FabricPC CNN with PC on "
            f"{splits.train_images.shape[0]} clean images.",
            flush=True,
        )
        started = time.perf_counter()
        trained = train(
            params=params,
            structure=structure,
            train_loader=loader,
            optimizer=optimizer,
            config={"num_epochs": cfg["epochs"]},
            rng_key=jax.random.PRNGKey(seed + 1),
            algorithm="pc",
            verbose=True,
        )
        params = trained.params
        train_time = time.perf_counter() - started
        print(f"Seed {seed}: PC training finished in {train_time:.1f}s.", flush=True)

        eval_batch_fn = _make_eval_batch(structure)
        print(
            f"Seed {seed}: encoding training images for prototype memory.", flush=True
        )
        embedding_outputs = []
        for start in range(0, splits.train_images.shape[0], cfg["eval_batch_size"]):
            _, embedding, _ = eval_batch_fn(
                params,
                jax.device_put(
                    splits.train_images[start : start + cfg["eval_batch_size"]],
                    device,
                ),
                jax.random.PRNGKey(seed + start),
            )
            embedding_outputs.append(embedding)
        z_train = jnp.concatenate(embedding_outputs, axis=0)
        memory, memory_labels, center = build_memory(
            z_train,
            splits.train_labels,
            per_class=cfg["prototypes_per_class"],
            center=cfg["center_prototypes"],
            seed=seed,
        )
        print(f"Seed {seed}: built frozen prototype memory.", flush=True)

        baseline_graph = structure
        baseline_eval_batch = eval_batch_fn
        hopfield_graph = build_pc_mnist_graph(
            prototypes=memory,
            center=center,
            beta=cfg["beta_values"][0],
            infer_steps=infer_steps,
            eta_infer=eta_infer,
        )
        hopfield_params = initialize_params(hopfield_graph, jax.random.PRNGKey(seed))
        _assert_same_parameter_layout(params, hopfield_params)

        best_beta = cfg["beta_values"][0]
        best_score = -float("inf")
        for beta_index, beta in enumerate(cfg["beta_values"], start=1):
            print(
                f"Seed {seed}: selecting beta ({beta_index}/{len(cfg['beta_values'])}, "
                f"beta={beta:g}).",
                flush=True,
            )
            candidate_graph = build_pc_mnist_graph(
                prototypes=memory,
                center=center,
                beta=beta,
                infer_steps=infer_steps,
                eta_infer=eta_infer,
            )
            candidate_eval_batch = _make_eval_batch(candidate_graph)
            scores = []
            for kind, level in cfg["beta_selection_sets"]:
                val_images = (
                    splits.val_images
                    if kind == "clean" and level == 0
                    else mb.make_corrupted_test_set(
                        splits.val_images,
                        seed=seed,
                        kind=kind,
                        level=level,
                    )
                )
                outputs = graph_outputs(
                    candidate_graph,
                    params,
                    candidate_eval_batch,
                    val_images,
                    batch_size=cfg["eval_batch_size"],
                    device=device,
                    prototypes=memory,
                    memory_labels=memory_labels,
                    center=center,
                    beta=beta,
                    num_updates=1,
                )
                metrics, _, _ = mb.summarize_outputs(
                    outputs, splits.val_labels, memory, memory_labels
                )
                scores.append(metrics["head_acc"])
            score = float(np.mean(scores))
            if score > best_score + 1e-12:
                best_score, best_beta = score, beta
        print(
            f"Seed {seed}: selected beta={best_beta:g} "
            f"(validation head accuracy={best_score:.4f}).",
            flush=True,
        )

        hopfield_graph = build_pc_mnist_graph(
            prototypes=memory,
            center=center,
            beta=best_beta,
            infer_steps=infer_steps,
            eta_infer=eta_infer,
        )
        hopfield_eval_batch = _make_eval_batch(hopfield_graph)
        grids = mb.noise_grids(cfg)
        experiment_kinds = (
            ("E1_clean", "clean", 0.0),
            *[(f"E3a_{level:g}", "gaussian", level) for level in grids["gaussian"]],
            *[
                (f"E3b_{level:g}", "salt_pepper", level)
                for level in grids["salt_pepper"]
            ],
            *[(f"E3c_{level:g}", "occlusion", level) for level in grids["occlusion"]],
        )
        seed_row: dict[str, Any] = {
            "seed": seed,
            "beta_star": best_beta,
            "train_time_s": train_time,
            "pc_infer_steps": infer_steps,
            "pc_eta_infer": eta_infer,
        }
        for experiment_index, (experiment, kind, level) in enumerate(
            experiment_kinds, start=1
        ):
            print(
                f"Seed {seed}: evaluating {experiment_index}/{len(experiment_kinds)} "
                f"{experiment}.",
                flush=True,
            )
            test_images = (
                splits.test_images
                if kind == "clean" or level == 0
                else mb.make_corrupted_test_set(
                    splits.test_images,
                    seed=seed,
                    kind=kind,
                    level=level,
                )
            )
            baseline_outputs = graph_outputs(
                baseline_graph,
                params,
                baseline_eval_batch,
                test_images,
                batch_size=cfg["eval_batch_size"],
                device=device,
                prototypes=memory,
                memory_labels=memory_labels,
                center=center,
                beta=best_beta,
                num_updates=1,
            )
            hopfield_outputs = graph_outputs(
                hopfield_graph,
                params,
                hopfield_eval_batch,
                test_images,
                batch_size=cfg["eval_batch_size"],
                device=device,
                prototypes=memory,
                memory_labels=memory_labels,
                center=center,
                beta=best_beta,
                num_updates=1,
            )
            for variant, outputs in (
                ("baseline", baseline_outputs),
                ("hopfield", hopfield_outputs),
            ):
                metrics, _, _ = mb.summarize_outputs(
                    outputs, splits.test_labels, memory, memory_labels
                )
                seed_row[f"{experiment}_{variant}"] = metrics["head_acc"]
                all_rows.append(
                    {
                        "seed": seed,
                        "experiment": experiment,
                        "variant": variant,
                        "kind": kind,
                        "level": level,
                        **metrics,
                    }
                )
        seed_summaries.append(seed_row)

    payload = {
        "config": {
            **cfg,
            "algorithm": "pc",
            "pc_infer_steps": infer_steps,
            "pc_eta_infer": eta_infer,
            "bridge_comparison": "identity vs frozen HopfieldMemoryNode",
        },
        "environment": mb.environment_info(device),
        "beta_selection": (
            "mean validation head_acc over configured clean and corruption sets"
        ),
        "seeds": seed_summaries,
        "rows": all_rows,
    }
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
        writer.writerows(all_rows)

    summary = [
        "# FabricPC PC MNIST Baseline vs Hopfield",
        "",
        "- Algorithm: `predictive coding (PC)`",
        "- Comparison: same trained FabricPC parameters; identity vs frozen memory",
        f"- PC inference: `{infer_steps}` steps at eta `{eta_infer}`",
        f"- Seeds: `{len(cfg['seeds'])}`",
        f"- Prototype memory: `{cfg['prototypes_per_class']} per class`, "
        f"centered `{cfg['center_prototypes']}`",
        "",
        "| Seed | Beta* | Clean baseline | Clean Hopfield | Gaussian 0.5 baseline | Gaussian 0.5 Hopfield |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summaries:
        summary.append(
            f"| {row['seed']} | {row['beta_star']:g} | "
            f"{row.get('E1_clean_baseline', 0.0):.4f} | "
            f"{row.get('E1_clean_hopfield', 0.0):.4f} | "
            f"{row.get('E3a_0.5_baseline', 0.0):.4f} | "
            f"{row.get('E3a_0.5_hopfield', 0.0):.4f} |"
        )
    (report_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"FabricPC PC MNIST reports saved to: {report_dir}", flush=True)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MNIST identity-versus-Hopfield comparison trained with FabricPC PC."
    )
    parser.add_argument("--config", type=Path, default=mb.DEFAULT_CONFIG_PATH)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--infer-steps", type=int, default=DEFAULT_INFER_STEPS)
    parser.add_argument("--eta-infer", type=float, default=DEFAULT_ETA_INFER)
    args = parser.parse_args(argv)
    if args.infer_steps <= 0:
        parser.error("--infer-steps must be positive.")
    if not math.isfinite(args.eta_infer) or args.eta_infer <= 0:
        parser.error("--eta-infer must be positive and finite.")

    cfg = mb.load_config(args.config, quick=args.quick)
    if args.reports_dir is not None:
        cfg["reports_dir"] = str(args.reports_dir)
    else:
        cfg["reports_dir"] = (
            "reports/mnist_pc_prototype_retrieval_quick"
            if args.quick
            else "reports/mnist_pc_prototype_retrieval"
        )
    _run_task(cfg, infer_steps=args.infer_steps, eta_infer=args.eta_infer)


if __name__ == "__main__":
    main()
