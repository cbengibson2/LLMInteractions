# =============================================================================
# run_selfcorrect_ollama.R
# Self-Correction alignment loop using local Ollama (llama3.2) — no API cost.
#
# Run from terminal:
#   Rscript ~/Projects/LLMREM/run_selfcorrect_ollama.R
#
# Prerequisites:
#   - Ollama running locally with llama3.2 pulled
#       ollama serve          (if not already running as a service)
#       ollama pull llama3.2
#   - conflict.Rdata and rem_empirical.Rdata on disk
# =============================================================================

cat("══ Self-Correction Ollama Run ══\n")
cat("Started:", format(Sys.time()), "\n\n")

# =============================================================================
# 1. PACKAGES
# =============================================================================

devtools::load_all("llrem", quiet = TRUE)
library(llrem)
library(survival)
library(ggplot2)

# =============================================================================
# 2. DATA & EMPIRICAL REM
# =============================================================================

STUDY_DIR       <- Sys.getenv("STUDY_DIR", ".")  # project root
data("edgelist", package = "llrem")
edgelist     <- prepare_edgelist(edgelist)
all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)
node_labels  <- if (exists("node_labels")) node_labels else NULL

cat("Nodes:", length(all_nodes), "| Events:", nrow(edgelist), "\n")

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

emp_rdata <- file.path(STUDY_DIR, "rem_empirical.Rdata")
if (file.exists(emp_rdata)) {
  load(emp_rdata)
  cat("Loaded cached empirical REM.\n")
} else {
  cat("Fitting empirical REM (no cache found)...\n")
  rem_empirical <- fit_rem(edgelist, formula = rem_formula,
                           n_nonevents = 200L, base_riskset = base_riskset)
  save(rem_empirical, file = emp_rdata)
}
cat("\nEmpirical coefficients:\n")
print(round(coef(rem_empirical), 3))

# =============================================================================
# 3. LOAD SELF-CORRECTION FUNCTIONS & BEHAVIORAL QUERY HELPERS
# =============================================================================

source(file.path(STUDY_DIR, "selfcorrection.R"))

# Load the full risk-set data frame (poss_edges) to compute empirical
# sufficient-statistic means for make_behavior_queries().
# Update this path to wherever conflict.Rdatadf.Rdata lives on your machine.
poss_edges_path <- file.path(STUDY_DIR, "conflict.Rdatadf.Rdata")
local({
  e <- new.env(parent = emptyenv())
  load(poss_edges_path, envir = e)
  poss_edges <<- e[[ls(e)[1]]]   # assign whatever object is in the file
})
cat("poss_edges loaded:", nrow(poss_edges), "rows\n")

stat_means <- make_stat_means(rem_formula, poss_edges)
behavior_query_str <- make_behavior_queries(stat_means)

cat("\nBehavioral guidelines to be used as correction context:\n")
cat(behavior_query_str, "\n")

# =============================================================================
# 4. PROVIDER & CONFIG
# =============================================================================

CONFLICT_CONTEXT <- paste(
  "You are participating in a simulation of an international militarized",
  "conflict network. Actors are sovereign countries. Each event represents",
  "one country using military force against another."
)

# Switch host to target a remote Ollama server, e.g. "192.168.1.65"
OLLAMA_HOST <- "192.168.1.65"

sc_cfg <- rem_cfg(
  provider       = llm_provider_ollama(model = "llama3.2:1b", host = OLLAMA_HOST),
  context        = CONFLICT_CONTEXT,
  action_verb    = "attacked",
  actor_noun     = "Country",
  node_labels    = node_labels,
  history_window = 60L
)

sc_checkpoint_dir <- file.path(STUDY_DIR, "checkpoints_selfcorrect_remote")
dir.create(sc_checkpoint_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 5. LOAD INITIAL SEQUENCE FROM STUDY3 CHECKPOINT
# ollama_predict_obs matches the query_factory used below (predict).
# =============================================================================

initial_sequence <- tryCatch({
  ckpt_path <- file.path(STUDY_DIR, "checkpoints_study3", "ollama_predict_obs.RData")
  env <- new.env(parent = emptyenv())
  load(ckpt_path, envir = env)
  seq_df <- rbind(
    edgelist[1L, c("time", "sender", "receiver", "eveid")],
    do.call(rbind, env$llm_rows)
  )
  row.names(seq_df) <- NULL
  cat(sprintf("Loaded initial sequence: %d events from %s\n",
              nrow(seq_df), basename(ckpt_path)))
  seq_df
}, error = function(e) {
  warning("Could not load initial sequence: ", conditionMessage(e),
          "\nIteration 1 will generate a new sequence.")
  NULL
})

# =============================================================================
# 6. RUN
# =============================================================================

sc_result <- selfcorrect_llm_rem(
  edgelist              = edgelist,
  formula               = rem_formula,
  cfg                   = sc_cfg,
  rem_empirical         = rem_empirical,
  query_factory         = make_query_predict,
  n_iterations          = 3L,
  convergence_tol       = 0.2,
  correction_tol        = 0.05,
  behavior_queries      = behavior_query_str,
  use_suffstats         = TRUE,
  emp_stat_means        = unlist(stat_means[stat_means$occurred == 1L,
                                             setdiff(names(stat_means), c("occurred", "RecencyMedian"))]),
  term_descriptions     = CONFLICT_TERM_DESCRIPTIONS,
  initial_sequence      = NULL,
  n_events              = 600L,
  n_nonevents           = 200L,
  base_riskset          = base_riskset,
  use_empirical_history = TRUE,
  rate_limit_secs       = 0,
  seed_base             = 3000L,
  checkpoint_dir        = sc_checkpoint_dir
)

cat(sprintf("\n══ Finished: %s after %d iteration(s). ══\n",
            if (sc_result$converged) "Converged" else "Did not converge",
            sc_result$n_iter_run))

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

out_rdata <- file.path(STUDY_DIR, "selfcorrect_ollama_results.RData")
save(sc_result, file = out_rdata)
cat("Results saved to:", out_rdata, "\n")

# =============================================================================
# 8. PLOTS
# =============================================================================

plots <- plot_convergence(sc_result, tol = 0.5)

ggsave(file.path(STUDY_DIR, "selfcorrect_ollama_rmse.pdf"),
       plots$rmse, width = 7, height = 4)
ggsave(file.path(STUDY_DIR, "selfcorrect_ollama_terms.pdf"),
       plots$terms, width = 9, height = 5)

cat("Plots saved to:", STUDY_DIR, "\n")
cat("Done:", format(Sys.time()), "\n")
