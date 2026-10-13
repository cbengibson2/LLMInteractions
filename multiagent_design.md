# Multi-Agent REM-LLM Simulation: Design Sketch

## Core Idea

The empirical REM defines a continuous-time hazard over all directed dyads.
At every moment the hazard surface encodes who is likely to act and against
whom, given the full event history. The multi-agent simulation uses this
surface to govern *when and by whom* events are initiated, while delegating
the *specific receiver choice* to an LLM agent representing the sender.

The REM and the LLM are doing different jobs:

| Component | Governed by |
|---|---|
| Who acts next (sender activation) | REM hazard marginal |
| Against whom (receiver selection) | LLM agent |
| Sequence dynamics (endogeneity) | REM updating on each new event |

---

## Step-by-Step Loop

### Initialisation

1. Fit (or load) the empirical REM on the MID edgelist. This gives
   $\hat{\boldsymbol{\beta}}$.
2. Precompute the base risk set for all $N$ actors.
3. Instantiate one LLM agent config (`rem_cfg`) per actor with its country
   identity and the shared conflict context.
4. Seed the running event history with the first empirical event by default.
   A `cold_start = TRUE` option omits the seed, beginning with a uniform
   hazard surface. Cold-start runs exhibit a burn-in period before endogenous
   structure accumulates.

---

### Per-Tick Logic

At tick $t$:

**Step 1 — Evaluate the hazard surface.**

Compute the Cox partial hazard for every directed dyad $(i,j)$, $i \neq j$:

$$h_{ij}(t) = \exp\!\left(\hat{\boldsymbol{\beta}}^\top \mathbf{x}_{ij}(t)\right)$$

where $\mathbf{x}_{ij}(t)$ are the REM sufficient statistics (Repetition,
RecInitiation, Recencylog, etc.) computed from the current history. This is
an $N \times (N-1)$ matrix of relative hazards.

**Step 2 — Derive sender activation probabilities.**

Marginalise over receivers to get each sender's total hazard mass:

$$H_i(t) = \sum_{j \neq i} h_{ij}(t)$$

Normalise to a probability distribution over senders:

$$p_i(t) = \frac{H_i(t)}{\sum_k H_k(t)}$$

**Step 3 — Sample the number of events this tick.**

Draw the number of simultaneous events $K_t$ from a Poisson with rate
$\lambda$, where $\lambda$ is estimated from the empirical inter-event
distribution (mean events per unit time). For the MID data this will
typically be 1 or 2 per tick.

**Step 4 — Sample $K_t$ senders.**

Draw $K_t$ senders without replacement from $\{1,\ldots,N\}$ weighted by
$p_i(t)$. High-hazard actors (recent aggressors, actors in active dyads)
are more likely to be activated.

**Step 5 — Query each activated agent.**

For each sampled sender $i$, call the LLM with:
- The sender's identity (`"You are Country i"`)
- The shared event history (last $W$ events)
- The list of valid targets

The LLM returns a receiver $j^*$. All $K_t$ queries can be dispatched
in parallel since they all condition on the same pre-tick history.
Co-activated senders do not see each other's choices within the tick;
within-tick events are invisible until tick $t+1$.

**Step 6 — Append events and update history.**

Add the $K_t$ new events to the shared log with timestamp $t$. All agents
see these events at tick $t+1$.

**Step 7 — Recompute sufficient statistics.**

Update $\mathbf{x}_{ij}(t+1)$ for the affected dyads. Only dyads
involving the senders or receivers from tick $t$ need updating — the rest
are unchanged.

---

## Why This Is Principled

**Sender activation is endogenous.** $p_i(t)$ is high for actors who have
recently initiated conflict, whose targets have retaliated, or who are in
active dyadic relationships — exactly the actors the empirical REM says are
at elevated hazard. It is not a fixed rate; it evolves as the sequence
unfolds.

**Receiver choice is identity-conditioned.** The LLM brings world knowledge
the REM doesn't have: which countries have historical rivalries, geographic
proximity, power asymmetries. The sender roleplay results show this is where
LLM performance is strongest.

**The two components are complementary.** The REM provides structural
discipline — preventing the sequence from drifting into statistically
unrealistic territory — while the LLM provides content that makes individual
choices interpretable and scenario-responsive.

**Drift is self-correcting.** If LLM receiver choices are structurally
inconsistent (e.g. targeting actors with no prior relationship when Repetition
is a strong predictor), the hazard surface at the next tick will reflect the
reduced endogenous momentum for those dyads. The REM penalises drift
automatically.

---

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| $\hat{\boldsymbol{\beta}}$ | Empirical REM coefficients | From fitted MID model |
| $W$ | History window shown to each agent | 30 events |
| $\lambda$ | Mean events per tick | Estimated from empirical inter-event times |
| $K_t$ | Events per tick (Poisson draw) | ~1–2 for MID |
| $N$ | Number of actors | 200 (MID: 95 active, 105 inactive) |
| `n_ticks` | Simulation length | User-specified |

---

## Implementation Plan

### New function: `run_multiagent_rem()`

**Arguments:**
- `edgelist` — empirical sequence (for REM fit and seeding)
- `rem_fit` — fitted coxph object (or fit internally)
- `cfgs` — named list of `rem_cfg` objects, one per actor
- `n_ticks` — number of simulation ticks
- `lambda` — Poisson event rate (estimated from data if NULL)
- `history_window` — events shown to each agent
- `base_riskset` — precomputed risk set
- `cold_start` — logical; if TRUE begin with uniform hazard rather than seeding from empirical event 1
- `seed` — RNG seed
- `checkpoint_path` — for resumable runs
- `cl` — optional parallel cluster for hazard evaluation

**Returns:**
- `$sequence` — data frame of generated events (time, sender, receiver)
- `$sender_probs` — $T \times N$ matrix of $p_i(t)$ at each tick
- `$n_events_per_tick` — vector of $K_t$

### New helper: `compute_hazard_surface()`

Takes the current history, base risk set, and $\hat{\boldsymbol{\beta}}$,
returns the $N \times (N-1)$ matrix of partial hazards. Reuses
`remstats_generate()` internals with `n_nonevents = 0L`.

### Additions to existing code

- `remstats_generate()` already computes sufficient statistics for realized
  events; `compute_hazard_surface()` calls it with the full risk set
- `make_query_roleplay_sender()` already exists and needs no changes
- Parallel dispatch reuses `parallel::mclapply` (already in Suggests)

---

## Evaluation

The same REM comparison framework from Study 3 applies directly. Fit a REM
to the generated multi-agent sequence and compare coefficients and sufficient
statistic means to the empirical baseline. The NTriadic sign and the
Recencylog magnitude are the primary diagnostics.

Additionally, the $p_i(t)$ matrix enables a new analysis: does sender
activation in the simulation track the empirical sender sequence? High
correlation between $p_i(t)$ and the empirical sender indicator at each
time step would confirm that the REM hazard is correctly identifying who acts
when.

---

## Open Design Questions

1. **Seeding.** Start from empirical event 1, or cold-start with an
   uninformative uniform hazard? Cold-start avoids conditioning on empirical
   data but produces a burn-in period before endogenous structure emerges.

2. **Simultaneous events and history.** When $K_t > 1$, all senders see the
   same pre-tick history. This is clean but means within-tick events don't
   influence each other. An alternative is to process activated senders
   sequentially within a tick in random order, updating history between each.

3. **LLM failure handling.** A parse failure occurs when `first_valid_int()`
   returns `NA_integer_` — three causes: (a) the API exhausts retries and
   returns `character(0)` or `NA_character_`; (b) the LLM responds in prose
   without a parseable integer; (c) the returned integer is not in
   `valid_targets` (e.g., own sender ID, out-of-range value). The fallback
   is to sample the receiver from the REM hazard row $h_{ij}(t)$ for
   sender $i$, weighted by $h_{ij}(t) / \sum_{j} h_{ij}(t)$. This is
   strictly better than uniform random sampling: the hazard-weighted draw
   reproduces the structural sparsity of the conflict network, favors
   historically active dyad partners, and rarely selects the 105 inactive
   countries (whose hazard mass is near zero for most senders).

4. **Scaling.** 95 actors, potentially several activations per tick, over
   thousands of ticks means many API calls. The Poisson rate $\lambda$ and
   the concentration of $p_i(t)$ on a few high-hazard actors will keep most
   ticks to 1–2 calls in practice — but this should be profiled.

5. **Endogenous frame vs. deliberative frame.** The proposed design does not
   prescribe the agent's reasoning frame. Both the 2×2 frame manipulation
   from the research proposal and the default deliberative frame can be tested
   within the same architecture by swapping the `cfgs` list.
