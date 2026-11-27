"""Train, checkpoint, and resume a small graph with a fixed Hopfield bank.

Run from the repository root with ``python examples/hopfield_fixed_bank.py``.
Pass ``--checkpoint-dir`` to keep the Orbax checkpoint after the example exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from fabricpc import setup_jax
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import GlobalStateInit, initialize_params
from fabricpc.hopfield import prepare_hopfield_bank
from fabricpc.nodes import HopfieldLinear, Linear
from fabricpc.training import evaluate, train

FEATURES = 2
BANK_CAPACITY = 3
SETTINGS = {
    "hopfield_strength": 0.75,
    "beta": 2.0,
    "eta_infer": 0.1,
    "infer_steps": 3,
    "latent_std": 0.2,
    "learning_rate": 0.01,
    "weight_decay": 0.2,
}


def make_training_objects(settings):
    source = Linear(shape=(FEATURES,), name="input")
    hidden = HopfieldLinear(
        shape=(FEATURES,),
        name="hidden",
        hopfield_strength=settings["hopfield_strength"],
        beta=settings["beta"],
        weight_init=NormalInitializer(std=0.3),
    )
    output = Linear(
        shape=(FEATURES,), name="output", weight_init=NormalInitializer(std=0.3)
    )
    structure = graph(
        nodes=[source, hidden, output],
        edges=[
            Edge(source=source, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=source, y=output),
        inference=InferenceSGD(
            eta_infer=settings["eta_infer"], infer_steps=settings["infer_steps"]
        ),
        graph_state_initializer=GlobalStateInit(
            initializer=NormalInitializer(std=settings["latent_std"])
        ),
    )
    optimizer = optax.adamw(
        learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    return structure, optimizer


def restore_checkpoint(directory: Path):
    """Restore the demo's known graph, AdamW, and bank layout from disk."""
    structure, optimizer = make_training_objects(SETTINGS)
    params = initialize_params(structure, jax.random.PRNGKey(0))
    # The template supplies types, shapes, and device placement. Its arrays
    # and scalar settings are independent of the checkpoint contents.
    target = {
        "params": params,
        "opt_state": optimizer.init(params),
        "node_buffers": {
            "hidden": {
                "hopfield_memory": jnp.zeros((BANK_CAPACITY, FEATURES)),
                "hopfield_active_mask": jnp.zeros((BANK_CAPACITY,), dtype=bool),
            },
        },
        "settings": {key: type(value)() for key, value in SETTINGS.items()},
        "rng_key": jax.random.PRNGKey(0),
        "next_epoch": 0,
    }
    restore_args = ocp.checkpoint_utils.construct_restore_args(target)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            directory.resolve(), item=target, restore_args=restore_args
        )

    bank = restored["node_buffers"]["hidden"]
    restored["node_buffers"]["hidden"] = prepare_hopfield_bank(
        bank["hopfield_memory"], active_mask=bank["hopfield_active_mask"]
    )
    return restored


def assert_same_tree(actual, expected):
    actual_leaves, actual_structure = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_structure = jax.tree_util.tree_flatten(expected)
    assert actual_structure == expected_structure
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
        assert np.asarray(actual_leaf).dtype == np.asarray(expected_leaf).dtype


def run_example(directory: Path) -> dict[str, float]:
    """Compare a restored next step with the same step in continuous training."""
    params_key, training_key = jax.random.split(jax.random.PRNGKey(42))
    structure, optimizer = make_training_objects(SETTINGS)
    params = initialize_params(structure, params_key)
    batches = [
        {
            "x": jnp.asarray([[0.7, 0.25], [-0.5, 1.0]]),
            "y": jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        }
    ]
    memory = jnp.asarray([[1.0, 0.25], [4.0, -2.0], [-0.5, 0.75]])
    node_buffers = {
        "hidden": prepare_hopfield_bank(memory, active_mask=[True, False, True]),
    }

    first_states = []
    first_epoch = train(
        params,
        structure,
        batches,
        optimizer,
        {"num_epochs": 1},
        training_key,
        node_buffers=node_buffers,
        iter_callback=lambda ctx: first_states.append(ctx.state),
        verbose=False,
    )
    checkpoint = {
        "params": first_epoch.params,
        "opt_state": first_epoch.opt_state,
        "node_buffers": node_buffers,
        "settings": dict(SETTINGS),
        "rng_key": training_key,
        "next_epoch": 1,
    }
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(directory.resolve(), checkpoint)
    restored = restore_checkpoint(directory)
    assert_same_tree(restored, checkpoint)

    resumed_structure, resumed_optimizer = make_training_objects(restored["settings"])
    resumed_states = []
    resumed = train(
        restored["params"],
        resumed_structure,
        batches,
        resumed_optimizer,
        {"num_epochs": 1},
        restored["rng_key"],
        opt_state=restored["opt_state"],
        start_epoch=restored["next_epoch"],
        node_buffers=restored["node_buffers"],
        iter_callback=lambda ctx: resumed_states.append(ctx.state),
        verbose=False,
    )

    reference_states = []
    reference = train(
        params,
        structure,
        batches,
        optimizer,
        {"num_epochs": 2},
        training_key,
        node_buffers=node_buffers,
        iter_callback=lambda ctx: reference_states.append(ctx.state),
        verbose=False,
    )
    assert_same_tree(first_states[-1], reference_states[0])
    assert_same_tree(resumed.params, reference.params)
    assert_same_tree(resumed.opt_state, reference.opt_state)
    assert_same_tree(resumed_states[-1], reference_states[-1])
    assert_same_tree(resumed_states[-1].nodes["hidden"].buffers, node_buffers["hidden"])
    assert resumed.iter_results == reference.iter_results[1:]
    return evaluate(
        resumed.params,
        resumed_structure,
        batches,
        {},
        restored["rng_key"],
        node_buffers=restored["node_buffers"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, help="Path for a new checkpoint."
    )
    args = parser.parse_args()
    setup_jax(platform="cpu")
    if args.checkpoint_dir is None:
        with TemporaryDirectory(prefix="hopfield-fixed-bank-") as scratch:
            metrics = run_example(Path(scratch) / "checkpoint")
    else:
        metrics = run_example(args.checkpoint_dir)
        print(f"Checkpoint saved to {args.checkpoint_dir.resolve()}")

    print("Restored bank and next training step match the uninterrupted run.")
    print(f"Evaluation energy {metrics['energy']:.6f}")


if __name__ == "__main__":
    main()
