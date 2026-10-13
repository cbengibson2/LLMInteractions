# =============================================================================
# run_wargame_oos.R
# Out-of-sample wargame scenarios using run_wargame().
#
# Seeds each scenario from the FULL empirical conflict history, then simulates
# N_TICKS forward. Three scenarios are run from the same seed point:
#
#   A. Baseline  -- no interventions; sequential move game
#   B. Military  -- actor 3 strikes actor 7 at tick 50; then free simulation
#   C. Peace     -- ceasefire between actors 3 and 7 (ticks 50-100) plus a
#                   mediator (actor 96) enters at tick 55
#
# After each scenario, a REM is fitted to the generated sequence and
# coefficients are compared against the empirical baseline.
# =============================================================================

library(survival)
library(ggplot2)
devtools::load_all("llrem", quiet = TRUE)

# =============================================================================
# 1. CREDENTIALS (AWS Bedrock via environment variables)
#    Set these in your shell or .Renviron -- do not hardcode here.
# =============================================================================

if (nzchar(Sys.getenv("AWS_ACCESS_KEY_ID"))) {
  message("Using AWS credentials from environment.")
} else {
  stop("AWS credentials not found. Set AWS_ACCESS_KEY_ID and ",
       "AWS_SECRET_ACCESS_KEY in your environment or .Renviron.", call. = FALSE)
}

# =============================================================================
# 2. DATA & EMPIRICAL REM
# =============================================================================

data("edgelist", package = "llrem")
edgelist <- prepare_edgelist(edgelist)

all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)

cat(sprintf("Nodes: %d  |  Empirical events: %d\n",
            length(all_nodes), nrow(edgelist)))

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

cat("\nFitting empirical REM...\n")
rem_empirical <- fit_rem(edgelist, rem_formula,
                         n            = length(all_nodes),
                         n_nonevents  = 200L,
                         base_riskset = base_riskset)

cat("\nEmpirical coefficients:\n")
print(round(coef(rem_empirical), 3))

# =============================================================================
# 3. SHARED CONFIGURATION
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

# Wargame config: sequential moves, endogenous framing
wgcfg <- wargame_cfg(
  sequential       = TRUE,
  behavioral_frame = "endogenous"
)

N_TICKS  <- 500L
CKPT_DIR <- file.path(STUDY_DIR,"checkpoints_wargame")
dir.create(CKPT_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 4. SCENARIO A — BASELINE (no interventions)
# =============================================================================

cat("\n=== SCENARIO A: Baseline post-sample continuation ===\n")

sim_A <- run_wargame(
  edgelist        = edgelist,
  rem_fit         = rem_empirical,
  cfg             = cfg,
  n_ticks         = N_TICKS,
  interventions   = list(),
  wgcfg           = wgcfg,
  lambda          = NULL,
  seed            = 42L,
  checkpoint_path = file.path(CKPT_DIR, "scenario_A.RData")
)

cat(sprintf("Scenario A: %d events generated over %d ticks.\n",
            nrow(sim_A$sequence), N_TICKS))

# =============================================================================
# 5. SCENARIO B — MILITARY INTERVENTION (actor 3 strikes actor 7 at tick 50)
# =============================================================================

cat("\n=== SCENARIO B: Military intervention (t=50, 3->7) ===\n")

sim_B <- run_wargame(
  edgelist        = edgelist,
  rem_fit         = rem_empirical,
  cfg             = cfg,
  n_ticks         = N_TICKS,
  interventions   = list(
    intervention_military(tick = 50L, sender = 3L, receiver = 7L,
                          label = "Strike_3v7_t50")
  ),
  wgcfg           = wgcfg,
  lambda          = NULL,
  seed            = 42L,
  checkpoint_path = file.path(CKPT_DIR, "scenario_B.RData")
)

cat(sprintf("Scenario B: %d events generated.\n", nrow(sim_B$sequence)))
cat("Intervention log:\n")
print(sim_B$intervention_log)

# =============================================================================
# 6. SCENARIO C — CEASEFIRE + THIRD-PARTY ENTRY
#    Ceasefire between 3<->7 for ticks 50-100; mediator (96) enters at tick 55
# =============================================================================

cat("\n=== SCENARIO C: Ceasefire (t=50-100) + Mediator entry (t=55) ===\n")

sim_C <- run_wargame(
  edgelist        = edgelist,
  rem_fit         = rem_empirical,
  cfg             = cfg,
  n_ticks         = N_TICKS,
  interventions   = list(
    intervention_ceasefire(
      tick_start = 50L, tick_end = 100L,
      dyads      = list(c(3L, 7L), c(7L, 3L)),
      label      = "Ceasefire_3v7"
    ),
    intervention_entry(
      tick           = 55L,
      actor_id       = 96L,
      actor_label    = "Mediator",
      initial_targets = c(3L, 7L),
      label          = "MediatorEntry_t55"
    )
  ),
  wgcfg           = wgcfg,
  lambda          = NULL,
  seed            = 42L,
  checkpoint_path = file.path(CKPT_DIR, "scenario_C.RData")
)

cat(sprintf("Scenario C: %d events generated.\n", nrow(sim_C$sequence)))
cat("Intervention log:\n")
print(sim_C$intervention_log)

# =============================================================================
# 7. POST-RUN REM COMPARISON
# =============================================================================

stat_terms <- c("Imitation", "Repetition", "RRecInitiation",
                "DyadRecencyLog", "IOviolation", "NTriadic")

fit_scenario_rem <- function(sim, label, n_nodes, formula, base_rs) {
  seq_data <- sim$sequence
  if (nrow(seq_data) < 100L) {
    message(label, ": too few events for stable REM fit (", nrow(seq_data), ").")
    return(NULL)
  }
  tryCatch(
    fit_rem(seq_data, formula, n = n_nodes, n_nonevents = 200L,
            base_riskset = base_rs),
    error = function(e) { message(label, " REM fit failed: ", e$message); NULL }
  )
}

rem_A <- fit_scenario_rem(sim_A, "A", length(all_nodes),
                           rem_formula, base_riskset)
rem_B <- fit_scenario_rem(sim_B, "B", length(all_nodes) + 1L,
                           rem_formula, base_riskset)
rem_C <- fit_scenario_rem(sim_C, "C", length(all_nodes) + 1L,
                           rem_formula, base_riskset)

# Coefficient comparison table
coef_list <- list(
  Empirical  = coef(rem_empirical),
  Scenario_A = if (!is.null(rem_A)) coef(rem_A) else rep(NA_real_, 6L),
  Scenario_B = if (!is.null(rem_B)) coef(rem_B) else rep(NA_real_, 6L),
  Scenario_C = if (!is.null(rem_C)) coef(rem_C) else rep(NA_real_, 6L)
)
coef_df <- do.call(cbind, lapply(coef_list, function(x) round(x[stat_terms], 3)))
cat("\n=== REM Coefficient Comparison ===\n")
print(coef_df)

# RMSE vs empirical
emp_coef <- coef(rem_empirical)[stat_terms]
for (label in c("Scenario_A", "Scenario_B", "Scenario_C")) {
  sc <- coef_list[[label]]
  if (!any(is.na(sc))) {
    rmse <- sqrt(mean((sc[stat_terms] - emp_coef)^2))
    cat(sprintf("RMSE (%s vs Empirical): %.3f\n", label, rmse))
  }
}

# =============================================================================
# 8. VISUALISATION — coefficient forest plot
# =============================================================================

coef_long <- do.call(rbind, lapply(names(coef_list), function(nm) {
  x <- coef_list[[nm]]
  data.frame(
    scenario = nm,
    term     = names(x[stat_terms]),
    estimate = as.numeric(x[stat_terms]),
    stringsAsFactors = FALSE
  )
}))
coef_long <- coef_long[!is.na(coef_long$estimate), ]

p_coef <- ggplot(coef_long, aes(x = estimate, y = term, colour = scenario,
                                 shape = scenario)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 3, position = ggplot2::position_dodge(width = 0.5)) +
  labs(title   = "REM coefficients: empirical vs wargame scenarios",
       x       = "Coefficient estimate",
       y       = NULL,
       colour  = NULL,
       shape   = NULL) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

print(p_coef)
ggsave(file.path(STUDY_DIR,"wargame_coef_comparison.pdf"),
       p_coef, width = 8, height = 5)

# =============================================================================
# 9. SEQUENTIAL MOVE INSPECTION
#    Show how within-tick events evolved for the first tick with >= 2 events
# =============================================================================

first_multi_tick <- which(sapply(sim_A$within_tick_sequences,
                                 function(x) !is.null(x) && nrow(x) >= 2L))[1L]
if (!is.na(first_multi_tick)) {
  cat(sprintf("\n--- Sequential move trace (Scenario A, tick %d) ---\n",
              first_multi_tick))
  print(sim_A$within_tick_sequences[[first_multi_tick]])
}

# =============================================================================
# 10. SAVE
# =============================================================================

save(sim_A, sim_B, sim_C,
     rem_empirical, rem_A, rem_B, rem_C,
     coef_df,
     file = file.path(STUDY_DIR,"wargame_oos_results.RData"))

cat("\nAll results saved to wargame_oos_results.RData\n")
