# Multi-Agent REM-LLM Conflict Simulation: Method and Preliminary Results

## 1. Background

Relational Event Models (REMs) fitted to the Militarized Interstate Dispute
(MID) dataset reveal strong endogenous structure in conflict sequences: states
repeat prior targets (Repetition), reciprocate recent attacks (RecInitiation),
and the hazard for active dyads decays sharply over time (Recencylog). Study 3
established that LLM agents, when assigned the identity of the empirical sender
and shown the observed event history, reproduce these statistics with striking
fidelity — Claude Haiku 4.5 achieves RMSE 0.309 across all six REM covariates
in the sender roleplay condition, with Repetition, RecInitiation, and
IOviolation all within rounding error of the empirical baseline.

Two persistent gaps remain across all single-agent conditions and both providers:

- **NTriadic** is positive in every LLM condition (Claude sender roleplay: 0.041;
  Llama sender roleplay: 0.476) while the empirical coefficient is strongly
  negative (−1.325), reflecting an implicit "enemy of my enemy is my enemy"
  heuristic in LLM training data that directly contradicts empirical conflict
  network dynamics.
- **Recencylog** is consistently less negative than the empirical baseline,
  indicating that LLMs spread events more evenly across the risk set than
  historical data warrants.

A reasoning probe administered to ten high-endogeneity events from the best
single-agent condition (Claude sender roleplay) revealed a dissociation between
narration and behavior: seven of ten events were classified as endogenously
framed, but in two cases the model explicitly argued for a novel target in free
text before producing the habitual endogenous choice as its integer response.
Both dissociation events involved the same sender (Country 3, events t=630 and
t=734), which verbally reasoned toward Country 1 on strategic grounds but
selected Country 19 in both the re-probed and original study responses.

---

## 2. Multi-Agent Architecture

The single-agent studies fix the sender at each step using the empirical sender
sequence. The multi-agent simulation removes this constraint: sender activation
is itself governed by the REM hazard surface, so the sequence evolves
autonomously from its own endogenous momentum rather than being scaffolded by
empirical ordering.

### 2.1 Per-Tick Loop

At each tick $t$:

**Step 1 — Hazard surface.** Compute the Cox partial hazard for every directed
dyad $(i, j)$, $i \neq j$:

$$h_{ij}(t) = \exp\!\left(\hat{\boldsymbol{\beta}}^\top \mathbf{x}_{ij}(t)\right)$$

where $\mathbf{x}_{ij}(t)$ are the six REM sufficient statistics (Imitation,
Repetition, RecInitiation, Recencylog, IOviolation, NTriadic) computed from the
running event history, and $\hat{\boldsymbol{\beta}}$ are the coefficients from
the empirical MID model.

**Step 2 — Sender activation probabilities.** Marginalise over receivers and
normalise:

$$H_i(t) = \sum_{j \neq i} h_{ij}(t), \qquad p_i(t) = \frac{H_i(t)}{\sum_k H_k(t)}$$

Actors with high recent activity and established dyadic relationships receive
elevated $p_i(t)$; actors with no conflict history remain near zero.

**Step 3 — Event count.** Draw $K_t \sim \text{Poisson}(\lambda)$ where
$\lambda$ is estimated from the empirical inter-event distribution.

**Step 4 — Sample senders.** Draw $K_t$ senders without replacement weighted
by $p_i(t)$.

**Step 5 — Query agents.** For each activated sender $i$, call the LLM with
the sender's country identity and the last 30 events as history. All $K_t$
queries condition on the same pre-tick history; within-tick events are not
visible to co-activated senders.

**Step 6 — Parse failure fallback.** If the LLM returns an unparseable
response, the receiver is drawn from the REM hazard row for that sender:
$h_{ij}(t) / \sum_j h_{ij}(t)$. This preserves structural plausibility and
naturally avoids selecting the many historically inactive actors.

**Step 7 — Update history.** Append the $K_t$ new events; recompute sufficient
statistics for affected dyads.

### 2.2 Relationship Between Components

| Component | Governed by |
|---|---|
| Who acts next (sender activation) | REM hazard marginals $p_i(t)$ |
| Against whom (receiver selection) | LLM agent identity roleplay |
| Sequence dynamics (endogeneity) | REM updating on each new event |

The REM provides structural discipline — preventing the sequence from drifting
into statistically implausible territory — while the LLM provides
identity-conditioned receiver choices that bring geopolitical content (rivalries,
power asymmetries, geographic proximity) that the REM does not represent. Parse
failures degrade gracefully to the REM-only choice rather than to uniform random
sampling, which would inflate NTriadic by spreading events across inactive actors.

---

## 3. Empirical Baseline

The empirical REM is fitted on 1,708 MID conflict events among 95 historically
active state actors. Coefficients and sufficient statistic means serve as the
evaluation target for all simulation conditions.

**Table 1. Empirical REM coefficients (Cox proportional hazards, MID dataset)**

| Covariate | Coefficient |
|---|---|
| Imitation | −0.207 |
| Repetition | 0.859 |
| RecInitiation | 3.410 |
| Recencylog | −0.344 |
| IOviolation | −4.640 |
| NTriadic | −1.325 |

**Table 2. Empirical sufficient statistic means**

| Covariate | Mean |
|---|---|
| Imitation | 0.278 |
| Repetition | 0.764 |
| RecInitiation | 0.919 |
| Recencylog | −5.461 |
| IOviolation | 0.058 |
| NTriadic | 0.001 |

---

## 4. Single-Agent Benchmark (Study 3)

**Table 3. Sufficient statistic RMSE by provider and condition (lower = better)**

| Provider | Condition | RMSE |
|---|---|---|
| Bedrock/Claude (Haiku 4.5) | Sender roleplay | **0.309** |
| Bedrock/Llama (3.1 70B) | Sender roleplay | 0.417 |
| Bedrock/Claude (Haiku 4.5) | Predict | 1.641 |
| Bedrock/Llama (3.1 70B) | Random roleplay | 2.228 |
| Bedrock/Claude (Haiku 4.5) | Random roleplay | 2.217 |

Sender roleplay substantially outperforms both predict and random roleplay
conditions for both providers. The accuracy improvement is driven by role
assignment rather than history observation alone: random roleplay also shows
the observed history but produces much weaker endogenous structure.

**Table 4. REM coefficients — sender roleplay vs empirical**

| Covariate | Empirical | Claude Sender | Llama Sender |
|---|---|---|---|
| Imitation | −0.207 | 0.033 | −0.594 |
| Repetition | 0.859 | 1.101 | 0.613 |
| RecInitiation | 3.410 | 2.894 | 3.902 |
| Recencylog | −0.344 | −0.397 | −0.341 |
| IOviolation | −4.640 | −4.458 | −4.213 |
| NTriadic | −1.325 | **0.041** | **0.476** |

Repetition, RecInitiation, Recencylog, and IOviolation are well recovered by
both providers. NTriadic sign reversal is the single most consistent failure
across all conditions and providers.

---

## 5. Multi-Agent Simulation: Preliminary Results

### 5.1 Setup

- Provider: Bedrock/Claude Haiku 4.5
- Ticks: 1,000
- Seed: empirical event 1 (non-cold-start)
- Lambda: estimated from empirical inter-event distribution (~1.02 events/tick)
- History window: 30 events
- REM coefficients: from fitted empirical model (Table 1)

### 5.2 Simulation Mechanics

The 1,000-tick run produced 1,023 events (mean 1.02/tick, max 6, 363 zero-ticks).
The Poisson rate estimated from the empirical sequence is low, reflecting the
bursty, sparse nature of MID conflicts. The sender activation surface at tick 1
(conditioned on the empirical seed event) concentrated 54% of probability mass
on a single actor (node 10) and 35% on a second (node 40); 81 of 95 actors had
$p_i < 0.001$.

As the simulation accumulated its own history, the activation surface evolved:
by the end of the run, nodes 38, 8, 82, 61, and 33 dominated sender activation,
reflecting whichever dyadic relationships the LLM-generated sequence happened to
establish. This demonstrates that sender activation is genuinely endogenous —
$p_i(t)$ responds to the LLM's own choices, not to a fixed schedule.

### 5.3 Sufficient Statistic Comparison

**Table 5. Sufficient statistic means — empirical vs multi-agent (1,000 ticks)**

| Covariate | Empirical | Multi-agent | Δ |
|---|---|---|---|
| Imitation | 0.278 | 0.484 | +0.206 |
| Repetition | 0.764 | 0.512 | −0.252 |
| RecInitiation | 0.919 | 0.848 | −0.071 |
| Recencylog | −5.461 | −4.606 | +0.855 |
| IOviolation | 0.058 | 0.299 | +0.241 |
| NTriadic | 0.001 | 0.151 | +0.150 |

**RMSE: 0.392** (vs 0.309 for Claude single-agent sender roleplay at 1,000 events).

RecInitiation and Recencylog are both well recovered. Recencylog at −4.606
(vs −5.461 empirical) is a notable improvement over single-agent predict and
random roleplay conditions, consistent with the expectation that REM-governed
sender activation — which concentrates activations on actors with established
conflict histories — enforces temporal recency structure that the LLM alone
does not generate.

### 5.4 REM Coefficient Comparison

**Table 6. REM coefficients — empirical vs multi-agent sequence**

| Covariate | Empirical | Multi-agent |
|---|---|---|
| Imitation | −0.207 | 0.074 |
| Repetition | 0.859 | 0.516 |
| RecInitiation | 3.410 | 3.897 |
| Recencylog | −0.344 | −0.275 |
| IOviolation | −4.640 | −2.427 |
| NTriadic | −1.325 | **−0.215** |

**The NTriadic coefficient is negative in the multi-agent simulation** (−0.215),
in contrast to all single-agent conditions where it is positive (Claude sender
roleplay: +0.041; Llama sender roleplay: +0.476). This is the most important
result from the preliminary run. The endogenous sender activation mechanism
appears to partially suppress the "enemy of my enemy" bias: because $p_i(t)$ is
elevated for actors already in active dyadic relationships, the simulation tends
to re-activate established conflict pairs rather than opening new triadic
relationships. The NTriadic sufficient statistic mean (0.151) remains above the
empirical value (0.001), indicating that the bias is attenuated but not
eliminated.

Repetition (0.516 vs empirical 0.859) and IOviolation (−2.427 vs −4.640) show
the largest remaining gaps in the coefficient comparison. Both are likely partly
attributable to sequence composition effects: with only 1,023 events and a
Poisson rate near 1, the history available to each agent at any given tick is
sparser than in the single-agent study, which ran continuous event sequences.

---

## 6. Summary and Next Steps

The multi-agent architecture produces sequences that partially close the gap
between single-agent LLM output and the empirical MID baseline, particularly on
the NTriadic coefficient — the one structural feature that all single-agent
conditions fail to recover. The mechanism is as expected: REM-governed sender
activation concentrates event production on historically active dyads, which
implicitly suppresses the triadic heuristic by rarely activating actors whose
only motivation for targeting a given receiver would be third-party enemy
relationships.

The remaining gaps (Repetition, IOviolation) suggest that approximately 1,000
ticks is not sufficient for the sequence to reach a stable statistical regime,
and that the low Poisson rate limits how quickly the history accumulates. The
next steps are:

1. **Longer runs.** Extend to 3,000–5,000 ticks to allow the sufficient
   statistic means to stabilize, and re-estimate the REM on a rolling basis to
   assess whether the coefficients converge.

2. **Behavioral frame manipulation.** Implement the 2×2 design from the
   research proposal: cross deliberative vs. endogenous agent framing with
   labeled (country names) vs. opaque (numeric IDs) identity. NTriadic is the
   primary diagnostic — the endogenous frame is predicted to push it further
   negative.

3. **Provider comparison.** Run the same 1,000-tick simulation with
   Bedrock/Llama to assess whether the NTriadic attenuation is provider-specific
   or a general property of the REM-governed architecture.

4. **Sender activation validation.** Correlate $p_i(t)$ with the empirical
   sender indicator to assess whether the REM hazard correctly identifies who
   acts when, independent of LLM receiver choice.
