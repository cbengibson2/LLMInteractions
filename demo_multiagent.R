# =============================================================================
# demo_multiagent.R
# End-to-end walkthrough of run_multiagent_rem().
#
# Steps:
#   1. Load data and fit the empirical REM
#   2. Build one shared rem_cfg
#   3. Run a short multi-agent simulation (50 ticks)
#   4. Inspect sender activation probabilities
#   5. Compare sufficient statistic means against the empirical baseline
#   6. (Optional) Fit a REM on the generated sequence
# =============================================================================

library(survival)
library(ggplot2)
devtools::load_all("llrem", quiet = TRUE)

# =============================================================================
# 1. CREDENTIALS (AWS Bedrock)
# =============================================================================

if (nzchar(Sys.getenv("AWS_ACCESS_KEY_ID"))) {
  message("Using AWS credentials from environment.")
} else {
  stop("AWS credentials not found. Set AWS_ACCESS_KEY_ID and ",
       "AWS_SECRET_ACCESS_KEY in your environment or .Renviron.", call. = FALSE)
}

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root

# =============================================================================
# 2. DATA
# =============================================================================

data("edgelist", package = "llrem")
edgelist <- prepare_edgelist(edgelist)

all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)

cat(sprintf("Nodes: %d  |  Events: %d\n", length(all_nodes), nrow(edgelist)))
# Expected: Nodes: 95 (actors present in at least one event)  |  Events: 1708

# =============================================================================
# 3. EMPIRICAL REM
# =============================================================================

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

cat("\nFitting empirical REM...\n")
rem_empirical <- fit_rem(edgelist, rem_formula,
                         n            = length(all_nodes),
                         n_nonevents  = 200L,
                         base_riskset = base_riskset)

cat("\nEmpirical REM coefficients:\n")
print(round(coef(rem_empirical), 3))

# =============================================================================
# 4. HAZARD SURFACE DEMO
# =============================================================================
# Before running agents, inspect what the hazard surface looks like on the
# first 100 empirical events.

cat("\n--- Hazard surface after 100 empirical events ---\n")

hz100 <- compute_hazard_surface(edgelist[1:100, ], base_riskset,
                                 coef(rem_empirical))

H_i <- tapply(hz100$hazard, hz100$sender, sum)
p_i <- H_i / sum(H_i)

cat("Top 10 most-activated senders (p_i):\n")
print(round(sort(p_i, decreasing = TRUE)[1:10], 4))

cat(sprintf("\nActors with p_i > 0.01: %d / %d\n",
            sum(p_i > 0.01, na.rm = TRUE), length(p_i)))
cat(sprintf("Actors with p_i < 0.001 (near-zero hazard): %d\n",
            sum(p_i < 0.001, na.rm = TRUE)))

# =============================================================================
# 5. REM_CFG
# =============================================================================

provider <- llm_provider_bedrock(
  model = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
  name  = "Bedrock/Claude (Haiku 4.5)"
)

CONFLICT_CONTEXT <- paste(
  "You are participating in a simulation of an international militarized",
  "conflict network. Actors are sovereign countries. Each event represents",
  "one country using military force against another."
)

cfg <- rem_cfg(
  provider       = provider,
  context        = CONFLICT_CONTEXT,
  action_verb    = "attacked",
  actor_noun     = "Country",
  node_labels    = NULL,
  history_window = 30L
)

print(cfg)

# =============================================================================
# 6. MULTI-AGENT SIMULATION
# =============================================================================
# 1000 ticks. Lambda is estimated from the empirical inter-event distribution.
# Expected: ~1-2 LLM calls per tick; ~1000-2000 total API calls.

N_TICKS <- 1000L
CKPT    <- file.path(STUDY_DIR,"checkpoints_multiagent", "demo_1000.RData")
dir.create(dirname(CKPT), showWarnings = FALSE)

cat(sprintf("\nRunning multi-agent simulation: %d ticks...\n", N_TICKS))

sim <- run_multiagent_rem(
  edgelist        = edgelist,
  rem_fit         = rem_empirical,
  cfg             = cfg,
  n_ticks         = N_TICKS,
  lambda          = NULL,          # estimated from edgelist
  base_riskset    = base_riskset,
  cold_start      = FALSE,         # seed with empirical event 1
  seed            = 42L,
  checkpoint_path = CKPT
)

cat(sprintf("\nSimulation complete.\n"))
cat(sprintf("  Generated events : %d\n", nrow(sim$sequence)))
cat(sprintf("  Ticks run        : %d\n", length(sim$n_events_per_tick)))
cat(sprintf("  Events per tick  : mean=%.2f  max=%d  zero-ticks=%d\n",
            mean(sim$n_events_per_tick),
            max(sim$n_events_per_tick),
            sum(sim$n_events_per_tick == 0L)))

# =============================================================================
# 7. SENDER ACTIVATION OVER TIME
# =============================================================================
# Plot p_i(t) for the top 5 senders across all ticks.

cat("\n--- Sender activation summary ---\n")

mean_pi <- colMeans(sim$sender_probs, na.rm = TRUE)
top5    <- names(sort(mean_pi, decreasing = TRUE))[1:5]

cat("Top 5 senders by mean p_i across simulation:\n")
print(round(sort(mean_pi, decreasing = TRUE)[1:5], 4))

# Build a long data frame for ggplot
pi_long <- do.call(rbind, lapply(top5, function(node) {
  data.frame(
    tick   = seq_len(nrow(sim$sender_probs)),
    p_i    = sim$sender_probs[, node],
    sender = paste("Country", node),
    stringsAsFactors = FALSE
  )
}))

p_activation <- ggplot(pi_long, aes(x = tick, y = p_i, colour = sender)) +
  geom_line(linewidth = 0.7, alpha = 0.8) +
  labs(title  = "Sender activation probabilities over simulation ticks",
       x      = "Tick",
       y      = expression(p[i](t)),
       colour = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

print(p_activation)
ggsave(file.path(STUDY_DIR,"demo_sender_probs.pdf"),
       p_activation, width = 9, height = 4)

# =============================================================================
# 8. SUFFICIENT STATISTIC COMPARISON
# =============================================================================

stat_terms <- c("Imitation", "Repetition", "RRecInitiation",
                "DyadRecencyLog", "IOviolation", "NTriadic")

cat("\n--- Sufficient statistic comparison ---\n")

emp_sm <- compute_seq_stat_means(edgelist, rem_formula, base_riskset)[stat_terms]

sim_sm <- tryCatch(
  compute_seq_stat_means(sim$sequence, rem_formula, base_riskset)[stat_terms],
  error = function(e) { message("suffstat error: ", e$message); NULL }
)

if (!is.null(sim_sm)) {
  tab <- rbind(
    data.frame(Source = "Empirical",  t(round(emp_sm, 3)), check.names = FALSE),
    data.frame(Source = "Multi-agent", t(round(sim_sm, 3)), check.names = FALSE)
  )
  print(tab, row.names = FALSE)
  rmse <- sqrt(mean((sim_sm - emp_sm)^2))
  cat(sprintf("\nRMSE vs empirical: %.3f\n", rmse))
}

# =============================================================================
# 9. OPTIONAL: FIT REM ON GENERATED SEQUENCE
# =============================================================================
# Comment out if the sequence is too short for stable Cox estimation.

if (nrow(sim$sequence) >= 200L) {
  cat("\nFitting REM on generated sequence...\n")
  rem_sim <- tryCatch(
    fit_rem(sim$sequence, rem_formula,
            n            = length(all_nodes),
            n_nonevents  = 200L,
            base_riskset = base_riskset),
    error = function(e) { message("REM fit failed: ", e$message); NULL }
  )

  if (!is.null(rem_sim)) {
    cat("\nCoefficient comparison (empirical vs multi-agent):\n")
    cf <- cbind(
      Empirical     = round(coef(rem_empirical), 3),
      `Multi-agent` = round(coef(rem_sim), 3)
    )
    print(cf)
  }
}

# =============================================================================
# 10. SAVE
# =============================================================================

save(sim, rem_empirical, emp_sm, sim_sm,
     file = file.path(STUDY_DIR,"demo_multiagent_results.RData"))
cat("\nResults saved to demo_multiagent_results.RData\n")
