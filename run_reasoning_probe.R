# =============================================================================
# run_reasoning_probe.R
# Reasoning probe: ask Claude to explain its choices at moments where it
# produced highly endogenous behaviour in the sender_obs condition.
#
# For each sampled event we replay the original context the model saw, then
# ask two questions in sequence:
#   1. Free-text: "Before answering, explain in 2-3 sentences WHY you are
#      choosing your target."
#   2. The original integer-only prompt, to confirm the choice is stable.
#
# This lets us see whether the model articulates endogenous mechanisms
# (habit, recency, reciprocity) or rationalises strategically, even when
# its statistical output was endogenously accurate.
# =============================================================================

devtools::load_all("llrem", quiet = TRUE)
library(survival)

# =============================================================================
# 1. CREDENTIALS & PROVIDER
# =============================================================================

if (nzchar(Sys.getenv("AWS_ACCESS_KEY_ID"))) {
  message("Using AWS credentials from environment.")
} else {
  stop("AWS credentials not found. Set AWS_ACCESS_KEY_ID and ",
       "AWS_SECRET_ACCESS_KEY in your environment or .Renviron.", call. = FALSE)
}

probe_provider <- llm_provider_bedrock(
  model = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
  name  = "Bedrock/Claude"
)

# =============================================================================
# 2. LOAD DATA & RESULTS
# =============================================================================

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root

data("edgelist", package = "llrem")
edgelist  <- prepare_edgelist(edgelist)
load(file.path(STUDY_DIR, "study3_results.RData"))

all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)
node_labels  <- if (exists("node_labels")) node_labels else NULL

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

CONFLICT_CONTEXT <- paste(
  "You are participating in a simulation of an international militarized",
  "conflict network. Actors are sovereign countries. Each event represents",
  "one country using military force against another."
)

cfg <- rem_cfg(
  provider       = probe_provider,
  context        = CONFLICT_CONTEXT,
  action_verb    = "attacked",
  actor_noun     = "Country",
  node_labels    = node_labels,
  history_window = 30L
)

# =============================================================================
# 3. IDENTIFY HIGH-ENDOGENEITY EVENTS
#
# Compute per-event sufficient statistics for the sender_obs sequence and
# select a diverse sample of moments where the LLM made a strongly endogenous
# choice: high Repetition, high RRecInitiation, or strongly negative DyadRecencyLog.
# =============================================================================

seq_sender <- results[["bedrock_claude_sender_obs"]]$sequence

pe <- llrem:::remstats_generate(seq_sender, n = length(all_nodes),
                                 n_nonevents  = 0L,
                                 base_riskset = base_riskset)
pe <- llrem:::add_survival_cols(pe)
pe$t <- seq_len(nrow(pe))   # event index (1-based, aligned with seq_sender rows 2+)

# Score each event by how endogenous it looks; higher = more endogenous.
# Use Repetition + RRecInitiation + abs(DyadRecencyLog) as a simple composite.
pe$endogeneity_score <- pe$Repetition +
                        pe$RRecInitiation +
                        abs(pe$DyadRecencyLog) / max(abs(pe$DyadRecencyLog), na.rm = TRUE)

# Sample across score quantiles to get variety: don't just take the top-N.
# Pick ~5 from top quartile, ~3 from mid, ~2 from lower-mid.
set.seed(42L)
q75 <- quantile(pe$endogeneity_score, 0.75, na.rm = TRUE)
q50 <- quantile(pe$endogeneity_score, 0.50, na.rm = TRUE)
q25 <- quantile(pe$endogeneity_score, 0.25, na.rm = TRUE)

top_events  <- pe[pe$endogeneity_score >= q75, ]
mid_events  <- pe[pe$endogeneity_score >= q50 & pe$endogeneity_score < q75, ]
low_events  <- pe[pe$endogeneity_score >= q25 & pe$endogeneity_score < q50, ]

sampled_events <- rbind(
  top_events[sample(nrow(top_events), min(5L, nrow(top_events))), ],
  mid_events[sample(nrow(mid_events), min(3L, nrow(mid_events))), ],
  low_events[sample(nrow(low_events), min(2L, nrow(low_events))), ]
)
sampled_events <- sampled_events[order(sampled_events$t), ]

cat(sprintf("Probing %d events (endogeneity scores: %.2f – %.2f)\n",
            nrow(sampled_events),
            min(sampled_events$endogeneity_score),
            max(sampled_events$endogeneity_score)))

# =============================================================================
# 4. PROBE FUNCTION
#
# For each selected event:
#   - Reconstruct the history visible at that moment (up to history_window)
#   - Ask for a free-text explanation first, then confirm the integer choice
# =============================================================================

probe_event <- function(t, cfg, seq_df, all_nodes) {

  # History available at time t (rows before event t in seq_df)
  # seq_df row 1 is the seed; event t corresponds to seq_df row t+1
  past <- seq_df[seq_len(t), ]     # rows 1..t  (history before this choice)
  emp_event <- seq_df[t + 1L, ]    # row t+1 = the choice made

  sender_id     <- emp_event$sender
  valid_targets <- setdiff(all_nodes, sender_id)

  history_str <- llrem:::format_history(
    utils::tail(past, cfg$history_window), cfg
  )

  sender_label <- llrem:::node_label(sender_id, cfg)
  target_str   <- paste(
    vapply(valid_targets,
           function(v) sprintf("%s (ID: %d)", llrem:::node_label(v, cfg), v),
           character(1L)),
    collapse = ", "
  )

  # --- Question 1: free-text reasoning ---
  msgs_reason <- list(
    list(role = "system", content = paste0(
      cfg$context,
      "\n\nYou are roleplaying as one actor in this network. ",
      "Given the event history, you must decide who to target next. ",
      "Before stating your choice, explain your reasoning in 2-3 sentences."
    )),
    list(role = "user", content = paste0(
      "=== EVENT HISTORY ===\n", history_str,
      "\n\n=== YOUR TURN ===\n",
      "You are ", sender_label, " (ID: ", sender_id, ").\n",
      "Valid targets: ", target_str, "\n\n",
      "First explain in 2-3 sentences WHY you are choosing your target, ",
      "then on the final line write ONLY the integer ID of your chosen target."
    ))
  )

  raw_reason <- llm_call(cfg$provider, msgs_reason, max_tokens = 200L,
                          temperature = 0.7)

  # --- Question 2: integer-only (original prompt) to check choice stability ---
  msgs_choice <- list(
    list(role = "system", content = paste0(
      cfg$context,
      "\n\nYou are roleplaying as one actor in this network. ",
      "Given the event history, decide who you will target next. ",
      "Reply with ONLY the integer ID of your target — no other text."
    )),
    list(role = "user", content = paste0(
      "=== EVENT HISTORY ===\n", history_str,
      "\n\n=== YOUR TURN ===\n",
      "You are ", sender_label, " (ID: ", sender_id, ").\n",
      "Valid targets: ", target_str, "\n",
      "Which actor do you target? Reply with the integer ID only."
    ))
  )

  raw_choice <- llm_call(cfg$provider, msgs_choice, max_tokens = 15L,
                          temperature = 0.7)

  # Parse the integer ID from the choice response
  chosen <- llrem:::first_valid_int(raw_choice, valid_targets)

  # Also try to extract the final integer from the reasoning response
  reasoned_id <- llrem:::first_valid_int(
    tail(strsplit(raw_reason, "\n")[[1L]], 1L),
    valid_targets
  )

  list(
    t              = t,
    sender         = sender_id,
    sender_label   = sender_label,
    actual_receiver         = emp_event$receiver,
    Repetition     = pe$Repetition[pe$t == t],
    RRecInitiation  = pe$RRecInitiation[pe$t == t],
    DyadRecencyLog     = pe$DyadRecencyLog[pe$t == t],
    reasoning      = raw_reason,
    reasoned_id    = reasoned_id,
    choice_id      = chosen
  )
}

# =============================================================================
# 5. RUN PROBES
# =============================================================================

probe_results <- vector("list", nrow(sampled_events))
rate_limit_secs <- 1

for (i in seq_len(nrow(sampled_events))) {
  t <- sampled_events$t[i]
  cat(sprintf("\n[%d/%d] Probing event t=%d  (Repetition=%.2f, RecInit=%.2f, DyadRecencyLog=%.2f)\n",
              i, nrow(sampled_events), t,
              sampled_events$Repetition[i],
              sampled_events$RRecInitiation[i],
              sampled_events$DyadRecencyLog[i]))

  probe_results[[i]] <- tryCatch(
    probe_event(t, cfg, seq_sender, all_nodes),
    error = function(e) {
      message("  Error at t=", t, ": ", conditionMessage(e))
      NULL
    }
  )

  if (!is.null(probe_results[[i]])) {
    r <- probe_results[[i]]
    cat(sprintf("  Sender: %s  |  Actual choice: %d  |  Re-probed choice: %d  |  Reasoned ID: %d\n",
                r$sender_label, r$actual_receiver, r$choice_id, r$reasoned_id))
    cat("  Reasoning:\n")
    cat(paste0("    ", gsub("\n", "\n    ", r$reasoning)), "\n")
  }

  Sys.sleep(rate_limit_secs)
}

# =============================================================================
# 6. SAVE & SUMMARISE
# =============================================================================

probe_results <- Filter(Negate(is.null), probe_results)
save(probe_results, sampled_events,
     file = file.path(STUDY_DIR, "probe_results.RData"))
cat(sprintf("\n%d probe results saved to %s\n",
            length(probe_results),
            file.path(STUDY_DIR, "probe_results.RData")))

# Classify reasoning by keyword presence
classify_reasoning <- function(text) {
  text_lc <- tolower(text)
  endogenous_terms <- c("history", "previous", "before", "past", "repeated",
                        "again", "already", "familiar", "prior", "last time",
                        "pattern", "recent", "retaliat", "reciproc")
  strategic_terms  <- c("power", "strateg", "weak", "strong", "alliance",
                        "interest", "opportun", "threat", "capabilit",
                        "econom", "geopolit", "leverage", "vulnerab")
  n_endo <- sum(sapply(endogenous_terms, grepl, x = text_lc))
  n_strat <- sum(sapply(strategic_terms, grepl, x = text_lc))
  if (n_endo > n_strat) "endogenous"
  else if (n_strat > n_endo) "strategic"
  else "mixed"
}

cat("\n=== REASONING CLASSIFICATION ===\n")
for (r in probe_results) {
  cls <- classify_reasoning(r$reasoning)
  cat(sprintf("t=%d  %s → %d  [%s]\n",
              r$t, r$sender_label, r$actual_receiver, cls))
}

tbl <- table(sapply(probe_results, function(r) classify_reasoning(r$reasoning)))
cat("\nSummary:\n")
print(tbl)
