Below is a condensed audit of the most salient non-compilation problems I found after *manual* inspection of **resolver.hpp / resolver.cpp**.
For each issue I give

* **Location** – the smallest fragment that unambiguously identifies the code,
* **Problem** – why it is wrong (implementation, logic, maths, undefined behaviour, etc.),
* **Fix** – the simplest correction that makes the code mathematically and/or logically sound.
  (“…” means irrelevant lines removed for brevity).

---

### 1  Unscented Kalman Filter (UKF) implementation

| Location                                                                                                         | Problem                                                                                                                                                                                                                                                                                                                                                                                                                                | Fix                                                                                                                                                                                        |                                            |                                                                                                                             |
| ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `full_ukf_t::predict()` – last line of the state update loop<br>`float rad = … fast_sincosf(rad, sp[3], sp[4]);` | **State vector inconsistency.**  `sp[3]`/`sp[4]` (and consequently the mean `x[3]`,`x[4]`) are *derived* variables (`sin(yaw)`, `cos(yaw)`).  After the UKF **update** step `x[0]` (yaw) changes, but the corresponding `x[3]`,`x[4]` are **never refreshed**, so the next call to `generate_sigma_points()` copies a *self-contradictory* state into the first sigma point.  That breaks the UKF assumptions and slowly inflates `P`. | Recompute the trigonometric components immediately after `x[0]` is wrapped in both `predict()` and `update()`:<br>`x[3] = std::sinf(DEG2RAD(x[0]));`<br>`x[4] = std::cosf(DEG2RAD(x[0]));` |                                            |                                                                                                                             |
| `full_ukf_t::generate_sigma_points()`<br>`float scale = std::sqrt(N + alpha * alpha * (N + kappa) - N);`         | **Equation written twice-removed.**  The scaling factor must be `√(N+λ)` with `λ = α² (N+κ)−N` ([fjp.github.io][1]).  The present expression reduces to `√(α² (N+κ))`, i.e. it is missing the *α* term when simplified.                                                                                                                                                                                                                | `float lambda = alpha * alpha * (N + kappa) - N;`<br>`float scale  = std::sqrt(N + lambda);`                                                                                               |                                            |                                                                                                                             |
| `full_ukf_t::update()` – construction of `P`                                                                     | The textbook formula is `P ← P − K S Kᵀ`.  The inner loop calculates `sum = Σₖₗ Kᵢₖ Sₖₗ Kⱼₗ` but stores it with `P[i][j] -= sum` **outside** the `k,l` loops – therefore `sum` is reused for all `(i,j)` pairs of the *current* row, corrupting covariance symmetry.                                                                                                                                                                   | Move the `sum` initialisation **inside** the `(j)` loop or accumulate directly into `P[i][j]`.<br>`for(int i…){ for(int j…){ float sum=0; … P[i][j]-=sum;}}`                               |                                            |                                                                                                                             |
| `matrix_inverse_2x2()`                                                                                           | No check that `det != 0` beyond ε; still returns an “inverse” when \`                                                                                                                                                                                                                                                                                                                                                                  | det                                                                                                                                                                                        | <ε`, silently injecting Inf/Nan into `P\`. | Return `false`/throw or enlarge ε to a domain-specific bound; make UKF skip the update on a singular innovation covariance. |

---

### 2  Particle filter

| Location                                                         | Problem                                                                                                                                                                                                  | Fix |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `measurement_noise.stddev()` is used as if it were a *variance*. | The Gaussian likelihood is `exp(−½ δ²/σ²)` where `σ` is the **standard deviation**.  The comment says *variance*, but the code correctly uses `stddev()`.  *Rename the variable* to avoid future errors. |     |

---

### 3  Time–frequency (jitter) analysis

| Location                                 | Problem                                                                                                                                                                                                                                              | Fix                                                                                                                                                                                                          |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `update_jitter_incremental_vectorized()` | (a) The function never **pushes** `newest` into `delta_history`, so variance/autocorr are computed on a stale window.<br>(b) The incremental variance update keeps growing `m2`, but never removes the oldest contribution ⇒ variance drifts upward. | Add `j.delta_history.push(newest);` at the start.<br>Maintain a circular buffer of the last *N* samples and recompute `mean,variance` from that buffer or use Welford with *removal*, not the naïve formula. |

---

### 4  Kolmogorov-Arnold Network (KAN)

#### 4.1 Edge evaluation

| Location                                                                                                  | Problem                                                                                                                                                                      | Fix                                                            |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `spline_edge::evaluate()` / `evaluate_with_basis_and_deriv()`<br>`int k = std::min(int(t), NUM_KNOTS-2);` | For `x == grid_max` we get `t = NUM_KNOTS−1`, `k = NUM_KNOTS−2`, **t ← t-k = 1**.  The rightmost B-spline segment is never visited and continuity at the boundary is broken. | `if (x >= grid_max) { k = NUM_KNOTS-2; t = 1.0f; } else { … }` |

#### 4.2 Gradient flow

| Location                                               | Problem | Fix |
| ------------------------------------------------------ | ------- | --- |
| `kan_resolver_t::update()` – layer-2 loop<br>\`\`\`cpp |         |     |
| for (int b=0;b<4;++b){                                 |         |     |

```
float grad = grad_output[o]*basis[b];
layer2_edges[h][o].coeffs[k+b] -= lr*grad;
/* L1 update */
```

}
float spline\_deriv\_sum = 0.f;
for (int b=0;b<4;++b)
spline\_deriv\_sum += layer2\_edges\[h]\[o].coeffs\[k+b]\*deriv\[b];
\`\`\` | **Wrong order of operations.**  `spline_deriv_sum` is calculated *after* the coefficients have just been changed, i.e. it uses *θ − η ∇θ* for the Jacobian of the **current** step.  This biases the hidden-layer gradient and violates the implicit Euler step used by SGD. | Compute `spline_deriv_sum` **before** the coefficient update or cache the old coefficients:<br>`float w_old = layer2_edges…; spline_deriv_sum += w_old*deriv[b];` |
\| Same function – layer-1 loop | The same error is repeated for layer 1. | Idem. |

#### 4.3 Missing use of the *hit* signal

| Location                                                   | Problem                                                                                                                        | Fix                                                                                                                    |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `kan_resolver_t::update(const …, int true_side, bool hit)` | The parameter `hit` is never used.  The call-site intends to reinforce successful predictions, but the gradient is unaffected. | Introduce a weight, e.g. `float sample_weight = hit ? 1.0f : 0.3f;` and multiply **all** gradients by `sample_weight`. |

#### 4.4 Adaptive grid range

| Location                                             | Problem                                                                                                                                                                     | Fix                                                                |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `adapt_grid_ranges()`<br>\`\`\`cpp                   |                                                                                                                                                                             |                                                                    |
| edge.grid\_min += rate\*(input - grid\_min);         |                                                                                                                                                                             |                                                                    |
| edge.grid\_max += rate\*(input - grid\_max);         |                                                                                                                                                                             |                                                                    |
| \`\`\`                                               | Both lines move *towards* the input value, so after enough iterations `grid_min ≈ grid_max`, degenerating the spline to a constant and causing division-by-zero in `dt_dx`. | Update only when the point lies **outside** the grid:<br>\`\`\`cpp |
| if (input < edge.grid\_min) edge.grid\_min = input;  |                                                                                                                                                                             |                                                                    |
| if (input > edge.grid\_max) edge.grid\_max = input;  |                                                                                                                                                                             |                                                                    |
| \`\`\`<br>and always keep `grid_max – grid_min ≥ ε`. |                                                                                                                                                                             |                                                                    |

---

### 5  Contextual-bandit covariance update

| Location                                                  | Problem                                                                                                                                                                             | Fix                                                                                               |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `linear_model_t::update()`<br>`cov[i][j] -= k[i] * x[j];` | Sherman-Morrison requires `C ← C − k xᵀ C`, **not** `k xᵀ`.  Omitting the trailing `C` breaks symmetry/PSD and makes the confidence bound (which the bandit relies on) meaningless. | Replace the loop with<br>`for (int i=0;i<D;++i) for(int j=0;j<D;++j) cov[i][j] -= k[i]*cov_x[j];` |

---

### 6  Miscellaneous

| Location                                                                    | Problem                                                                                                                                                                                              | Fix                        |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `ring_buffer::push()`                                                       | The post-increment expression is longer than needed and branches unpredictably.                                                                                                                      | `index = (index + 1) % N;` |
| `update_jitter_cwt_vectorized()` – AVX2 path<br>`_mm256_load_ps(&ψ_re[n]);` | `ψ_re`/`ψ_im` are only **32-byte** aligned, but `_mm256_load_ps` requires 32-byte alignment.  They are declared with `alignas(32)`, **OK**.  If portability is desired, switch to `_mm256_loadu_ps`. |                            |

---

## 7  Summary of obligatory fixes for the KAN part

1. Guard the right-edge knot case in `evaluate*`.
2. Compute `spline_deriv_sum` **before** updating any coefficients.
3. Actually use the `hit` flag to scale the learning signal.
4. Re-write `adapt_grid_ranges` so that *only* the violated bound is moved and the interval never collapses.
5. (Optional but recommended) rename variables/arrays to avoid accidental shadowing of `basis`/`deriv` in nested scopes.

These corrections bring the implementation in line with the reference KAN formulations where
*edges carry cubic B-spline functions*, **nodes are linear**, and training uses plain SGD with L₁ sparsity ([GitHub][2], [Daily Dose of Data Science][3], [ML without tears][4], [Daniel Bethell][5]).

Applying the table above should make the resolver numerically stable and the KAN learner behave as intended.

[1]: https://fjp.at/blog/ukf?utm_source=chatgpt.com "Unscented Kalman Filter - fjp.github.io"
[2]: https://github.com/Mattral/Kolmogorov-Arnold-Networks?utm_source=chatgpt.com "TensorFlow Implementation of Kolmogorov-Arnold Network (KAN)"
[3]: https://www.dailydoseofds.com/a-beginner-friendly-introduction-to-kolmogorov-arnold-networks-kan/?utm_source=chatgpt.com "A Beginner-friendly Introduction to Kolmogorov Arnold Networks (KAN)"
[4]: https://mlwithouttears.com/2024/05/15/a-from-scratch-implementation-of-kolmogorov-arnold-networks-kan/?utm_source=chatgpt.com "A from-scratch implementation of Kolmogorov-Arnold Networks ..."
[5]: https://daniel-bethell.co.uk/posts/kan/?utm_source=chatgpt.com "Demystifying Kolmogorov-Arnold Networks: A Beginner-Friendly ..."
