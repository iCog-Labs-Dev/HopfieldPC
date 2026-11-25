# Modern Hopfield Step 1 JAX core

Step 1 implements the math for retrieving from a supplied memory bank. Given a batch of query vectors and a bank of stored vectors, the core produces a weighted average of the memories, evaluates the energy that matches this update, repeats the update with a fixed bank, and measures how concentrated the retrieval weights are. The four commits build these capabilities in order.

## Architecture and API

| Commit | Capability | Main implementation | Cumulative focused test cases |
| --- | --- | --- | ---: |
| 1 · `0e9899d` | [One-step retrieval and validation](#commit-1-one-step-retrieval-and-validation) | [Input validation](../../fabricpc/hopfield/_validation.py) and [retrieval](../../fabricpc/hopfield/functional.py) | 23 |
| 2 · `3ed3f88` | [Matching Hopfield energy](#commit-2-matching-hopfield-energy) | [Energy](../../fabricpc/hopfield/energy.py) | 34 |
| 3 · `b24bade` | [Fixed-bank multi-step retrieval](#commit-3-fixed-bank-multi-step-retrieval) | [Retrieval loop](../../fabricpc/hopfield/functional.py) | 46 |
| 4 · `a81045e` | [Mask-aware attention entropy](#commit-4-mask-aware-attention-entropy) | [Diagnostics](../../fabricpc/hopfield/diagnostics.py) | 55 |

The shared numerical operation is the masked, centered dot product. Retrieval applies softmax to those scores and takes a weighted average. Energy uses the log of the exponential sum of the same scores and restores the shift used for centering. Multi-step retrieval feeds the weighted average back as the next query. Entropy takes the weights as input. [Public exports](../../fabricpc/hopfield/__init__.py)

~~~mermaid
flowchart TD
    Inputs["Query, memory bank, active mask"] --> Host["Host validation for concrete inputs"]
    Host --> Prepare["JAX input preparation"]
    Prepare --> Scores["Raw dot products and centered masked scores"]
    Scores --> Weights["Softmax weights"]
    Scores --> Partition["Log partition"]
    Prepare --> Quadratic["Quadratic query term"]
    Partition --> Energy["Hopfield energy"]
    Quadratic --> Energy
    Weights --> Average["Weighted average of memory vectors"]
    Prepare -->|"Prepared bank"| Average
    Weights --> Entropy["Attention entropy"]
    Average -->|"Next query in multi-step retrieval"| Scores
~~~

| Symbol or call | Contract |
| --- | --- |
| $Q$, $M$ | Queries have shape `[B, D]`. Memory vectors are rows of a bank with shape `[N, D]`. Here $B$ is batch size, $D$ is feature size, and $N$ is bank capacity. Each size is positive. |
| `active_mask` | A shared boolean vector of shape `[N]`. The set $A$ contains the active indices and $K = \lvert A \rvert$ is the active count. Omitting the mask activates all $N$ rows. |
| `validate_hopfield_inputs` | Checks concrete inputs before a JAX transformation. Requires $K \ge 1$, finite queries, and finite active memory rows. |
| `hopfield_retrieve` | Returns retrieved vectors `[B, D]` and attention weights `[B, N]`. |
| `hopfield_energy` | Returns one energy per query `[B]` by default or a scalar with `reduce="mean"`. |
| `multi_step_retrieve` | Returns the final vectors and the weights that produced them. With `return_states=True`, it also returns a list of $T+1$ arrays of shape `[B, D]`, including the initial query. |
| `attention_entropy` | Takes attention weights `[B, N]` and returns one diagnostic value per query `[B]`. |

Query and memory share a floating dtype. Float16 and bfloat16 inputs are promoted before arithmetic, and the results stay float32. Float32 and float64 retain their dtype. To use float64, enable JAX x64 before constructing the arrays. Under `jax.jit`, keep `beta`, `include_constants`, `reduce`, `num_updates`, `return_states`, and `normalize` static wherever the corresponding function accepts them. The arrays and mask are runtime inputs.

## Commit 1 One-step retrieval and validation

### Calculation and intuition

For one query $q$ and active memory rows $m_i$, retrieval computes

$$
s_i = q^\top m_i,\qquad
p_i(q) = \frac{\exp(\beta s_i)}{\sum_{j\in A}\exp(\beta s_j)},\qquad
R(q) = \sum_{i\in A} p_i(q)m_i.
$$

The inverse temperature $\beta$ controls how strongly similarity differences affect the weights. For active memories $i$ and $j$, their weight ratio is $\exp(\beta(s_i-s_j))$. As $\beta$ approaches zero, the update approaches the mean of the active memories. Larger $\beta$ favors the highest scores. Identical scores receive identical weights at any $\beta$. The scores are raw dot products, so vector magnitudes as well as alignment affect them.

Because the weights are nonnegative and sum to one, $R(q)$ lies in the convex hull of the active memories. Intermediate weights naturally produce a blend. With a single active row the update returns that row for every query.

### Implementation flow

1. `prepare_inputs` checks ranks, positive dimensions, feature compatibility, dtypes, and the mask. It prepares the working dtype and replaces inactive bank rows with zeros before either matrix product. This allows inactive storage to contain `NaN` or `Inf` without affecting retrieval.
2. `validate_beta` checks that `beta` is a finite positive Python real in the positive normal range of the working dtype. The public host validator also reads array values to check finiteness and require an active row. Call it on concrete inputs before entering `jit` or `grad`. The JAX kernels check static metadata during tracing and rely on these runtime preconditions.
3. `_similarities` computes $QM^\top$. For each query, `_centered_scores` takes the maximum active similarity $a=\max_{i\in A}s_i$ and forms $\beta(s_i-a)$ on active slots. It sets inactive logits to negative infinity. Subtracting the same value from each active similarity preserves the softmax, and centering before multiplication keeps the largest scaled logit at zero. JAX treats the shift as a constant when differentiating the scores.
4. `_retrieve_prepared` applies softmax across bank slots and multiplies the resulting weights by the prepared bank. Inactive slots have exactly zero weight. Differentiation remains available with respect to the query and active memory values.

### Tests introduced

The suite uses an independent float64 NumPy implementation of the equations. Its small fixture includes an all-zero query, which gives tied similarities. The x64 fixture scopes that precision setting to each test. [Focused tests](../../tests/test_hopfield_core.py)

| Test | Purpose and assertion |
| --- | --- |
| `test_retrieval_matches_independent_numpy_reference` | Establish the raw-dot-product update at $\beta=0.25,2,12$. Compare retrieved vectors and weights with NumPy, and check nonnegative weights and row sums of one. Commit 2 extends this case to energy and renames it. |
| `test_single_memory_is_returned_for_every_query` | Verify the one-row limit. At two beta values every output is the single memory and every weight is one. Commit 2 adds the corresponding energy check. |
| `test_duplicate_memories_receive_equal_weights` | Verify tie handling. Three identical memories each receive one third of the weight at three beta values, and their weighted average is the same memory. |
| `test_nonfinite_inactive_values_do_not_affect_retrieval` | Compare a padded bank containing inactive `NaN` and `Inf` entries with its compact active bank. Host validation succeeds, active results agree, and inactive weights are exactly zero. |
| `test_retrieval_supports_query_and_active_memory_gradients` | Differentiate retrieval at a tied query with an inactive `NaN` row. The query gradient is 2. The two active memory gradients are 0.5 each and the inactive gradient is zero. |
| `test_compiled_and_vmapped_retrieval_matches_eager_results` | Check that masked eager batches, compiled batches, and a per-query `vmap` give the same retrieved states and weights. Commit 2 extends this case to energy and renames it. |
| `test_low_precision_promotes_before_dot_products` | Use float16 and bfloat16 vectors with entries of magnitude 300. Check float32 results and the expected one-hot retrieval. Commit 2 adds a float32 energy assertion for the same large query. |
| `test_standard_precision_output_dtype_is_preserved` | Check that float32 and float64 outputs retain their input dtype. Commit 2 adds energy to the dtype check. |
| `test_centering_prevents_positive_logit_overflow_at_large_beta` | With float32, similarities 4 and 3, and $\beta=10^{38}$, check that centering yields weights `[1, 0]` and returns the first memory. |
| `test_beta_must_be_positive_finite_and_representable` | Reject a boolean, zero, a negative value, NaN, infinity, and values above or below the positive normal float32 range. Commit 2 adds the energy API to the same checks. |
| `test_validator_rejects_invalid_shapes_dtypes_and_masks` | Reject malformed query or memory ranks, zero batch size or bank capacity, mismatched feature sizes or dtypes, integer data, and wrong mask shape or dtype. |
| `test_validator_rejects_empty_bank_and_nonfinite_active_data` | Reject an all-false mask and `NaN` or `Inf` in a query or an active memory row. |
| `test_validator_rejects_traced_inputs` | Calling host validation under `jax.jit` raises the error that directs callers to validate outside JAX transformations. |

## Commit 2 Matching Hopfield energy

### Calculation and intuition

Define the log partition $F$ in similarity units and the default energy $E_0$ by

$$
F(q) = \frac{1}{\beta}\log\sum_{i\in A}\exp(\beta q^\top m_i),
\qquad
E_0(q) = \frac12\lVert q\rVert^2 - F(q).
$$

The quadratic term penalizes large query norms. The log partition is a smooth maximum over the similarities, so subtracting it lowers energy in directions aligned with memories. Both terms depend on the query and appear in the default energy.

With `include_constants=True`, the function adds $\log(K)/\beta + \tfrac12\max_{i\in A}\lVert m_i\rVert^2$. These terms shift the energy by the same amount for every query at fixed bank, mask, and beta. The offset can change with the bank. For $K=1$, the full expression simplifies to $\tfrac12\lVert q-m_1\rVert^2$. The batch mean applies a factor $1/B$ to each query's energy gradient.

### Why retrieval follows the energy gradient

Differentiating the log partition with respect to a similarity $s_i$ gives $p_i(q)$. The derivative of $q^\top m_i$ with respect to $q$ is $m_i$, so the chain rule gives

$$
\nabla_q F(q) = \sum_{i\in A}p_i(q)m_i = R(q),
\qquad
\nabla_q E_0(q) = q - R(q).
$$

The update $q \leftarrow R(q)$ is therefore a unit gradient step on this energy. Its descent bound comes from the convexity of $F$. For any candidate $x$, convexity gives $F(x) \ge F(q) + R(q)^\top(x-q)$. Inserting that lower bound into $E_0(x)$ gives a quadratic upper bound on the energy.

$$
E_0(x) \le \frac12\lVert x\rVert^2 - F(q) - R(q)^\top(x-q).
$$

This bound is tight at $x=q$ and minimized at $x=R(q)$. The drop in the quadratic upper bound is $\tfrac12\lVert q-R(q)\rVert^2$, which proves

$$
E_0(q)-E_0(R(q)) \ge \frac12\lVert q-R(q)\rVert^2.
$$

This is an exact-arithmetic statement for the standalone update with fixed $M$, $A$, and $\beta$. The full energy has the same decrease because its offset cancels. At a stationary point $q=R(q)$, the update leaves the query at that point.

### Implementation flow and numerical limits

1. `hopfield_energy` prepares inputs and uses the same dot-product and centering helpers as retrieval. With $z_i=\beta(s_i-a)$, the default partition evaluates $a+\operatorname{logsumexp}(z)/\beta$. Adding back $a$ restores the absolute similarity scale.
2. The full energy moves the $\log K$ term into the partition and subtracts it before division by $\beta$. Its partition is $a+\beta^{-1}\log[(1/K)\sum_{i\in A}\exp(z_i)]$. When the centered log mean is close to zero, the implementation uses `log1p(mean(expm1(z)))` over active slots to preserve small differences. When all active centered scores are within machine epsilon of zero, it uses the first-order limit of the normalized partition, the mean active similarity. This avoids losing tiny scaled scores before division. The maximum squared memory norm is added to the energy separately.
3. The partition defines a custom Jacobian-vector product, or JVP, so that the derivative is the softmax-weighted sum of similarity changes. The factor $\beta$ cancels analytically with $1/\beta$. This keeps the gradient computable even when $1/\beta$ would underflow at large beta. Inactive rows have zero influence through both the partition and the prepared bank.

### Tests introduced or extended

| Test | Purpose and assertion |
| --- | --- |
| `test_retrieval_and_energy_match_independent_numpy_reference` | Extend the first retrieval fixture. At three beta values, compare the default energy with NumPy and check the full energy against the reference plus the analytical offset. |
| `test_single_memory_is_returned_for_every_query` | Add the exact one-memory identity for the full energy and compare it with half the squared distance to that memory. |
| `test_energy_mask_matches_compact_bank_in_value_and_memory_gradient` | For both energy modes, compare masked and compact banks. Check equal energies, equal active gradients, and exact zero inactive gradients. For the sum of query energies, check the default bank gradient $-P^\top Q$ and the extra $B m_i$ on the unique maximum-norm row when the offset is included. |
| `test_energy_query_gradient_is_query_minus_retrieved` | Check the defining identity for both energy modes, with a mask and a tied-score query. Compare eager and JIT gradients with `query - retrieved`. |
| `test_energy_mean_reduction_scales_query_gradient` | Check per-query and scalar shapes, the mean value, and the $1/B$ gradient factor. Also reject an unsupported reduction and a nonboolean `include_constants` flag. |
| `test_fixed_bank_retrieval_satisfies_energy_descent_bound` | For three beta values, a seeded four-query batch, and a masked bank, run five updates. Check the sufficient-decrease inequality per query at every step in both energy modes, with a floating-point tolerance. |
| `test_compiled_and_vmapped_core_matches_eager_results` | Extend transformation parity to the full energy. Check the eager, JIT, and per-query `vmap` paths against the same reference outputs. |
| `test_low_precision_promotes_before_dot_products` | Add the quadratic energy check. A query with two entries of 300 and zero memory has energy 90,000, which must be represented in float32. |
| `test_standard_precision_output_dtype_is_preserved` | Add energy to the float32 and float64 dtype assertions. |
| `test_energy_gradient_remains_finite_at_large_beta` | At $\beta=10^{38}$ in float32, check a compiled energy of $-3.5$ and a query gradient of $-3$. This catches the underflow that can remove the partition contribution to the gradient. |
| `test_full_energy_remains_finite_and_accurate_at_small_beta` | In float32, test a symmetric 128-row bank at $\beta=10^{-5}$ and $1.2\times10^{-38}$. Compare the full energy with its analytical small-beta approximation $9.5-4.5\beta$. |
| `test_beta_must_be_positive_finite_and_representable` | Apply the seven existing invalid-beta cases to the energy API as well as retrieval and host validation. |

## Commit 3 Fixed-bank multi-step retrieval

### Calculation and implementation flow

A fixed bank means the memory vectors, active mask, and beta are reused throughout the loop. The query and its weights evolve. With initial query $q_0$, define $p_t$ as the weights that produce the state $q_t$.

$$
p_t=p(q_{t-1}),\qquad q_t=\sum_{i\in A}p_{t,i}m_i=R(q_{t-1}),
\qquad t=1,\ldots,T.
$$

The function prepares the bank and mask once. A `jax.lax.scan` runs exactly `num_updates` steps with a carry of the latest state and its producing weights. The initial weights are zero placeholders, and the first step replaces them. The default $T=1$ matches the one-step API.

The return value is `(q_T, p_T)`. In particular, $p_T$ scores the penultimate state $q_{T-1}$. Scoring $q_T$ would produce the weights for $q_{T+1}$. With `return_states=True`, the scan emits each new state and the function prepends $q_0$, giving `[q0, ..., qT]` in the working dtype. With the default flag, the forward scan produces only the final carry and skips the trajectory output.

Every iteration uses the same energy from Commit 2, so its descent bound applies at each transition. The calculation can settle toward a self-consistent weighted combination of memories. The bank stays fixed during the forward calculation. Gradients through all steps with respect to the initial query and active bank values remain available.

### Tests introduced

| Test | Purpose and assertion |
| --- | --- |
| `test_multistep_default_matches_one_step` | Check that the default returns the one-step state and weights. The optional one-update trajectory starts at the input query and ends at the retrieved state. |
| `test_multistep_trajectory_and_final_weights_match_independent_reference` | Compare each state and the final weights after three updates with repeated NumPy calculations. Check that weights from scoring the final state differ on this fixture and that disabling the trajectory preserves the final outputs. |
| `test_compiled_multistep_matches_compact_active_bank` | Under JIT, run a padded bank containing an inactive nonfinite row. Check that its trajectory, final vector, and active weights match those of the compact bank, and that its inactive weight is exactly zero. |
| `test_multistep_gradients_match_unrolled_retrieval` | Compare compiled gradients through the scan with gradients from three explicit one-step calls. Check both trajectory settings, both query and memory gradients, and exact zero gradient for the inactive row. |
| `test_multistep_retains_float32_for_all_returned_states` | For float16 and bfloat16 inputs, check that the initial query, subsequent states, final output, and weights are all float32 and agree exactly with running the already-converted inputs. |
| `test_multistep_requires_a_positive_integer_update_count` | Reject a boolean, zero, a negative integer, and a noninteger count. |
| `test_multistep_requires_static_configuration` | Passing a traced update count or trajectory flag raises an error identifying the relevant static argument. |

## Commit 4 Mask-aware attention entropy

### Calculation and intuition

Shannon entropy measures how evenly the weights are distributed across active memories. Using natural logarithms gives entropy in nats for each query.

$$
H(p) = -\sum_{i\in A} p_i\log p_i.
$$

A zero weight contributes zero by continuity. A one-hot distribution has entropy zero, and the uniform distribution attains the maximum $\log K$. Set `normalize=True` to compute $H/\log K$ for $K>1$, which puts distributions over different active counts on a common zero-to-one scale up to roundoff. For $K=1$, the function defines normalized entropy as zero.

| Weights over two active memories | Entropy in nats | Normalized entropy |
| --- | ---: | ---: |
| `[1.0, 0.0]` | 0 | 0 |
| `[0.9, 0.1]` | 0.32508 | 0.46900 |
| `[0.5, 0.5]` | 0.69315 | 1 |

Entropy describes weight concentration. Retrieval accuracy requires a separate measure tied to the intended target.

### Implementation flow

1. `attention_entropy` checks the shape and dtype of the supplied weights and prepares the mask. The active entries are expected to be finite, nonnegative probabilities that sum to one per query. The function consumes the probabilities as supplied and uses the working dtype shared by the other core operations.
2. It replaces inactive entries with zero before computing the logarithms. At zero probability, the implementation evaluates the logarithm at one so that the term contributes zero and autodiff stays finite at the boundary. For $p_i>0$, the derivative of the entropy term is $-\log p_i-1$.
3. It sums over the bank dimension, giving one value per query. In normalized mode it divides by the log of the **active** count and uses a safe denominator for the single-active-memory case. Pass the same mask to retrieval and entropy so the active counts agree.

### Tests introduced

| Test | Purpose and assertion |
| --- | --- |
| `test_attention_entropy_matches_known_distributions` | Compare the one-hot, skewed, and uniform distributions in the table with analytical values for both raw and normalized entropy. |
| `test_attention_entropy_ignores_inactive_slots_and_uses_active_count` | Mask out `NaN` and `Inf` weights. Check raw entropy of $\log 2$, normalized entropy of one for two active uniform weights, and normalized entropy of zero for one active memory. |
| `test_attention_entropy_gradient_is_masked_and_finite_at_zero` | Check the analytic derivative at positive weights, exact zero gradient at an inactive entry, and a finite autodiff value at a zero active probability. |
| `test_compiled_attention_entropy_tracks_multistep_retrieval_concentration` | Pass final weights from compiled two-step retrieval into normalized entropy. Higher beta gives lower entropy for the asymmetric query in the fixture, while the tied query stays uniform at both beta values. |
| `test_attention_entropy_uses_working_precision` | Compare the uniform distribution with $\log 2$ in all four supported dtypes. Float16 and bfloat16 give float32 output, and float32 and float64 preserve dtype. |
| `test_attention_entropy_rejects_invalid_static_inputs` | Reject rank or size errors, invalid mask shape or dtype, integer weights, and a traced `normalize` flag. |

## Worked example from query to entropy

Use $q_0=(0.8,0.5)$, $\beta=5$, and the bank rows $(1,0),(0,1),(-1,0)$. All three rows are active. The raw similarities are $(0.8,0.5,-0.8)$. Subtracting 0.8 and multiplying by 5 gives logits $(0,-1.5,-8)$. Their softmax gives

$$
p_1 \approx (0.8173503,\, 0.1823755,\, 0.0002742),\qquad
q_1 \approx (0.8170761,\, 0.1823755).
$$

The first coordinate of $q_1$ is $p_{1,1}-p_{1,3}$, because the first and third memories point in opposite directions. The second coordinate is $p_{1,2}$. The energy gradient at $q_0$ is approximately $(-0.0170761,\,0.3176245)$. The first update lowers the energy by 0.0795567, above the lower bound of 0.0505885 from the squared move.

~~~python
import jax
import jax.numpy as jnp

from fabricpc.hopfield import (
    attention_entropy,
    hopfield_energy,
    hopfield_retrieve,
    multi_step_retrieve,
    validate_hopfield_inputs,
)

query = jnp.asarray([[0.8, 0.5]], dtype=jnp.float32)
bank = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=jnp.float32)
mask = jnp.asarray([True, True, True])
beta = 5.0
validate_hopfield_inputs(query, bank, beta=beta, active_mask=mask)

first, first_weights = hopfield_retrieve(query, bank, beta=beta, active_mask=mask)
run = jax.jit(
    multi_step_retrieve, static_argnames=("beta", "num_updates", "return_states")
)
final, final_weights, states = run(
    query, bank, beta=beta, num_updates=3, active_mask=mask, return_states=True
)
energies = [hopfield_energy(q, bank, beta=beta, active_mask=mask) for q in states]
_, second_weights = hopfield_retrieve(states[1], bank, beta=beta, active_mask=mask)
entropies = [
    attention_entropy(weights, active_mask=mask, normalize=True)
    for weights in (first_weights, second_weights, final_weights)
]

print("First state and weights", first, first_weights)
print("Final state and producing weights", final, final_weights)
print("Energy from q0 through q3", jnp.concatenate(energies))
print("Normalized entropy for updates 1 through 3", jnp.concatenate(entropies))
~~~

| State | Query vector, rounded | Default energy | Normalized entropy of weights producing the state |
| --- | --- | ---: | ---: |
| $q_0$ | $(0.8000000, 0.5000000)$ | −0.3953375 | — |
| $q_1$ | $(0.8170761, 0.1823755)$ | −0.4748942 | 0.4345886 |
| $q_2$ | $(0.9592930, 0.0401642)$ | −0.5003874 | 0.1556094 |
| $q_3$ | $(0.9898708, 0.0099941)$ | −0.5013934 | 0.0516013 |

The descent bound concerns energy. The entropy values in this example show the attention concentrating on the first memory as the iterations progress.

## Verification and integration boundary

The focused suite has 32 test functions and 55 cases after parametrization. The Step 1 audit at `a81045e` used Python 3.12.3 and JAX and jaxlib 0.12.0.1 on CPU. It recorded 85 passing tests across the Hopfield core, node-contract, external-node, and existing energy suites. The full repository run had 725 passing tests and six skips. The separate two-device sharding run passed all seven of its tests, including the six skipped in the standard run. Cross-backend PyTorch parity was deferred under the Step 1 plan. [Focused tests](../../tests/test_hopfield_core.py) · [CI workflow](../../.github/workflows/test.yml)

From the repository root with the development dependencies installed, run

~~~bash
python -m pytest tests/test_hopfield_core.py -q
python -m pytest tests/test_hopfield_core.py tests/test_node_contract.py tests/test_external_custom_node.py tests/test_energy.py -q
python -m pytest -q
XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m pytest tests/test_sharding.py -q
~~~

Step 2 will evaluate $\lambda E_0(z;M)$ at the current latent state $z$ of a FabricPC hidden node. The added latent gradient is $\lambda(z-R(z))$. The FabricPC solver owns the inference loop and combines this gradient with the existing predictive-coding terms. The integration must carry a fixed bank through inference, exclude it from optimizer updates, and reproduce baseline behavior when $\lambda=0$. The standalone multi-step loop is available for studying retrieval trajectories independently. [FabricPC baseline protocol](baseline_protocol.md)
