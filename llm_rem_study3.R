# =============================================================================
# llm_rem_study3.R
# Multi-Provider, Multi-Condition LLM vs Empirical REM Comparison
#
# Five conditions are run for each LLM provider:
#
#   1. predict_obs   – LLM predicts next sender AND receiver; observed history
#   2. random_obs    – LLM roleplays a random country;        observed history
#   3. sender_obs    – LLM roleplays the empirical sender;    observed history
#   4. random_llm    – LLM roleplays a random country;        LLM-generated history
#   5. sender_llm    – LLM roleplays the empirical sender;    LLM-generated history
#
# Conditions 1–3 (use_empirical_history = TRUE) isolate each generation strategy
# from compounding sequence drift. Conditions 4–5 (use_empirical_history = FALSE)
# let errors accumulate, producing an independent LLM sequence.
#
# All conditions are compared against an empirical REM.
# =============================================================================

#library(llrem)
library(survival)
library(ggplot2)
devtools::load_all("llrem", quiet = TRUE)

# =============================================================================
# 1. PROVIDER SELECTION
#
# Set ACTIVE_PROVIDERS to any subset of the names below. Only the listed
# providers will be run. Comment out or remove names to skip them.
#
#   "openai"       – OpenAI GPT (direct API key)
#   "anthropic"    – Anthropic Claude (direct API key)
#   "gemini"       – Google Gemini (direct API key)
#   "grok"         – xAI Grok (direct API key)
#   "ollama"       – Local/remote Ollama instance (no key needed)
#   "bedrock_claude" – Anthropic Claude via AWS Bedrock
#   "bedrock_llama"  – Meta Llama via AWS Bedrock
# =============================================================================

ACTIVE_PROVIDERS <- c(
  # "openai",
  # "anthropic",
  # "gemini",
  # "grok",
  # "ollama",
  "bedrock_claude",
  "bedrock_llama"
)

# =============================================================================
# 2. CREDENTIALS
# Fill in only the keys for providers you are using.
# AWS credentials apply to both bedrock_claude and bedrock_llama.
# Leave unused keys as empty strings ("").
# =============================================================================

# Set API keys in your shell environment or .Renviron -- do not hardcode here.
# AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
# apply to both bedrock_claude and bedrock_llama.
# Per-provider keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY.

# =============================================================================
# 3. PROVIDER DEFINITIONS
# Add or modify model strings here without touching the run logic below.
# Bedrock model IDs: enable each model in the AWS Console under
#   Amazon Bedrock > Model access before first use.
# =============================================================================

ALL_PROVIDERS <- list(

  openai = list(
    provider    = llm_provider_openai(model = "gpt-4o"),
    rate_limit  = 2
  ),

  anthropic = list(
    provider    = llm_provider_anthropic(model = "claude-haiku-4-5"),
    rate_limit  = 2
  ),

  gemini = list(
    provider    = llm_provider_gemini(model = "gemini-2.5-flash"),
    rate_limit  = 5   # free tier: 15 RPM
  ),

  grok = list(
    provider    = llm_provider_grok(model = "grok-3"),
    rate_limit  = 2
  ),

  ollama = list(
    provider    = llm_provider_ollama(model = "llama3.2"),  # run: ollama pull llama3.2
    rate_limit  = 0
  ),

  bedrock_claude = list(
    provider    = llm_provider_bedrock(
                    model  = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    name   = "Bedrock/Claude"
                  ),
    rate_limit  = 1   # Bedrock throttles per-account; tune as needed
  ),

  bedrock_llama = list(
    provider    = llm_provider_bedrock(
                    model  = "us.meta.llama3-1-70b-instruct-v1:0",
                    name   = "Bedrock/Llama"
                  ),
    rate_limit  = 1
  )
)

# Validate selection and build active lists
unknown <- setdiff(ACTIVE_PROVIDERS, names(ALL_PROVIDERS))
if (length(unknown) > 0L)
  stop("Unknown provider(s) in ACTIVE_PROVIDERS: ",
       paste(unknown, collapse = ", "))

providers   <- lapply(ALL_PROVIDERS[ACTIVE_PROVIDERS], `[[`, "provider")
rate_limits <- vapply(ALL_PROVIDERS[ACTIVE_PROVIDERS], `[[`, numeric(1L),
                      "rate_limit")
names(providers)   <- ACTIVE_PROVIDERS
names(rate_limits) <- ACTIVE_PROVIDERS

cat("Active providers:\n")
for (p in providers) print(p)

# =============================================================================
# 4. LOAD DATA
# =============================================================================

data("edgelist", package = "llrem")
edgelist <- prepare_edgelist(edgelist)

all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)

cat("Nodes:", length(all_nodes), "| Events:", nrow(edgelist), "\n")

# =============================================================================
# 5. REM FORMULA  (node_labels loaded with conflict.Rdata if available)
# =============================================================================

node_labels <- if (exists("node_labels")) node_labels else NULL

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

# =============================================================================
# 6. EMPIRICAL REM
# =============================================================================

STUDY_DIR      <- Sys.getenv("STUDY_DIR", ".")  # project root

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
# 7. CONDITIONS
# Each row defines one (query_factory, use_empirical_history) combination.
# =============================================================================

conditions <- list(
  list(
    id            = "predict_obs",
    label         = "Predict (obs. history)",
    query_factory = make_query_predict,
    use_emp_hist  = TRUE,
    seed          = 1101L
  ),
  list(
    id            = "random_obs",
    label         = "Random roleplay (obs. history)",
    query_factory = make_query_roleplay_random,
    use_emp_hist  = TRUE,
    seed          = 2202L
  ),
  list(
    id            = "sender_obs",
    label         = "Sender roleplay (obs. history)",
    query_factory = make_query_roleplay_sender,
    use_emp_hist  = TRUE,
    seed          = 3303L
  )#,
  #list(
   # id            = "random_llm",
  #  label         = "Random roleplay (LLM history)",
  #  query_factory = make_query_roleplay_random,
  #  use_emp_hist  = FALSE,
  #  seed          = 4404L
  #),
  #list(
  #  id            = "sender_llm",
  #  label         = "Sender roleplay (LLM history)",
  #  query_factory = make_query_roleplay_sender,
  #  use_emp_hist  = FALSE,
  #  seed          = 5505L
  #)
)

# =============================================================================
# 8. RUN ALL PROVIDERS × CONDITIONS
# Checkpoints are written every 10 events; delete to restart from scratch.
# =============================================================================


checkpoint_dir <- file.path(STUDY_DIR, "checkpoints_study3")
dir.create(checkpoint_dir, showWarnings = FALSE, recursive = TRUE)

results <- list()   # keyed by "<provider>_<condition_id>"

# -----------------------------------------------------------------------------
# 8a-recover. RECOVER FROM CHECKPOINTS
# Run this block to rebuild `results` from any checkpoint files already on
# disk — useful after an interrupted run or a session restart. Complete
# sequences (all events present) and partial ones are both recovered; partial
# ones will be resumed automatically by the collection loop below.
# -----------------------------------------------------------------------------

ckpt_files <- list.files(checkpoint_dir, pattern = "\\.RData$", full.names = TRUE)
n_ev        <- 1000L

for (f in ckpt_files) {
  key <- sub("\\.RData$", "", basename(f))

  # Don't overwrite a result that was already recovered or collected this session
  if (!is.null(results[[key]]$sequence)) next

  tryCatch({
    env <- new.env(parent = emptyenv())
    load(f, envir = env)
    n_rows <- length(env$llm_rows)

    seq_df <- rbind(edgelist[1L, c("time", "sender", "receiver", "eveid")],
                    do.call(rbind, env$llm_rows))
    row.names(seq_df) <- NULL

    results[[key]] <- list(
      sequence  = seq_df,
      responses = do.call(rbind, env$llm_rows)
    )

    status <- if (n_rows >= n_ev - 1L) "complete" else
      sprintf("partial (%d/%d)", n_rows, n_ev - 1L)
    message("Recovered ", key, " [", status, "]")
  }, error = function(e) {
    warning("Could not recover ", key, ": ", conditionMessage(e), call. = FALSE)
  })
}
results <- results[lapply(results, function(x) nrow(x[[2]])) >= n_ev - 1L]
cat(sprintf("%d / %d sequences recovered from checkpoints.\n",
            sum(vapply(results, \(r) !is.null(r$sequence), logical(1L))),
            length(providers) * length(conditions)))

# -----------------------------------------------------------------------------
# 8a. SEQUENCE COLLECTION  (hits the APIs)
# Sequences are checkpointed every 10 events. Re-running this loop after an
# interruption will resume from the last checkpoint automatically.
# fit = FALSE defers model fitting until all sequences are collected.
# -----------------------------------------------------------------------------

for (pname in names(providers)) {

  cfg <- rem_cfg(
    provider       = providers[[pname]],
    context        = paste(
      "You are participating in a simulation of an international militarized",
      "conflict network. Actors are sovereign countries. Each event represents",
      "one country using military force against another."
    ),
    action_verb    = "attacked",
    actor_noun     = "Country",
    node_labels    = node_labels,
    history_window = 30L
  )

  for (cond in conditions) {

    key <- paste0(pname, "_", cond$id)
    cat("\n=== Collecting:", pname, "|", cond$id, "===\n")

    tryCatch({
      results[[key]] <- run_llm_rem(
        edgelist              = edgelist,
        formula               = rem_formula,
        query_fn              = cond$query_factory(cfg),
        n_events              = 1000L,
        n_nonevents           = 200L,
        base_riskset          = base_riskset,
        seed                  = cond$seed,
        use_empirical_history = cond$use_emp_hist,
        checkpoint_path       = file.path(checkpoint_dir, paste0(key, ".RData")),
        rate_limit_secs       = rate_limits[[pname]],
        fit                   = FALSE
      )
      cat("Sequence collected:", key, "\n")
    }, error = function(e) {
      warning("Skipping ", key, ": ", conditionMessage(e), call. = FALSE)
    })
  }
}

cat("\nAll sequences collected. Saving before fitting...\n")
save(results, file = file.path(STUDY_DIR, "study3_sequences.RData"))

# -----------------------------------------------------------------------------
# 8b. MODEL FITTING  (local computation, no API calls)
# Safe to re-run independently after loading study3_sequences.RData.
# -----------------------------------------------------------------------------

cat("\nFitting REMs on collected sequences...\n")

for (key in names(results)) {
  if (is.null(results[[key]])) next
  if (!is.null(results[[key]]$rem)) next          # already fitted (e.g. reloaded)

  cat("  Fitting:", key, "\n")
  tryCatch({
    results[[key]]$rem <- fit_rem(
      results[[key]]$sequence,
      formula      = rem_formula,
      n_nonevents  = 200L,
      base_riskset = base_riskset
    )
  }, error = function(e) {
    warning("fit_rem failed for ", key, ": ", conditionMessage(e), call. = FALSE)
  })
}


# =============================================================================
# 9. ASSEMBLE MODEL LIST FOR COMPARISON
# =============================================================================

# Drop any NULL slots (failed conditions) before assembling models
results <- Filter(Negate(is.null), results)

# Helper: split a result key into provider display name and condition id.
# Keys are "<pname>_<cid>" where pname may contain underscores (e.g.
# "bedrock_claude"). Provider key is matched against ACTIVE_PROVIDERS;
# display name is taken from the provider object's $name field.
split_key <- function(key) {
  pkey  <- ACTIVE_PROVIDERS[startsWith(key, paste0(ACTIVE_PROVIDERS, "_"))]
  if (length(pkey) != 1L) stop("Cannot resolve provider for key: ", key)
  cid   <- substring(key, nchar(pkey) + 2L)
  pname <- providers[[pkey]]$name
  list(pname = pname, cid = cid)
}

# Full model list: empirical + all provider-condition combinations
all_models <- c(
  list(Empirical = rem_empirical),
  lapply(results, \(r) r$rem)
)

# Human-readable labels for each model
model_labels <- c(
  "Empirical",
  vapply(names(results), function(key) {
    p    <- split_key(key)
    cond <- Filter(\(c) c$id == p$cid, conditions)[[1L]]
    paste0(p$pname, "\n", cond$label)
  }, character(1L))
)

names(all_models) <- model_labels
results<-results[unlist(lapply(results,function(x) x$rem$iter))!=21]
save(results,file=file.path(STUDY_DIR,"checkpoint.Rdata"))
# =============================================================================
# 10. FOREST PLOTS — one per condition, all providers vs. empirical
# =============================================================================

condition_ids    <- vapply(conditions, `[[`, character(1L), "id")
condition_labels <- vapply(conditions, `[[`, character(1L), "label")

for (i in seq_along(conditions)) {
  cid   <- condition_ids[i]
  clbl  <- condition_labels[i]

  # Pick the empirical model plus all provider results for this condition
  cond_keys <- grep(paste0("_", cid, "$"), names(results), value = TRUE)
  cond_model_labels <- vapply(cond_keys, function(key) {
    split_key(key)$pname
  }, character(1L))

  cond_models <- c(
    list(Empirical = rem_empirical),
    setNames(lapply(results[cond_keys], \(r) r$rem), cond_model_labels)
  )

  compare_rem_models(
    models = cond_models,
    title  = paste0("Condition: ", clbl)
  )
}

# =============================================================================
# 11. COMBINED FACETED FOREST PLOT
# =============================================================================

extract_coefs <- function(model, source, condition) {
  s <- summary(model)$coefficients
  data.frame(
    term      = rownames(s),
    coef      = s[, "coef"],
    lower95   = s[, "coef"] - 1.96 * s[, "se(coef)"],
    upper95   = s[, "coef"] + 1.96 * s[, "se(coef)"],
    source    = source,
    condition = condition,
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

# Empirical row appears in every condition panel for reference
emp_rows <- do.call(rbind, lapply(condition_labels, \(cl)
  extract_coefs(rem_empirical, "Empirical", cl)
))

llm_rows <- do.call(rbind, lapply(names(results), function(key) {
  if (is.null(results[[key]])) return(NULL)
  p    <- split_key(key)
  clbl <- condition_labels[match(p$cid, condition_ids)]
  extract_coefs(results[[key]]$rem, p$pname, clbl)
}))

coef_df <- rbind(emp_rows, llm_rows)

term_order <- c("Imitation", "Repetition", "RRecInitiation",
                "DyadRecencyLog", "IOviolation", "NTriadic")
coef_df$term      <- factor(coef_df$term,      levels = rev(term_order))
coef_df$condition <- factor(coef_df$condition, levels = condition_labels)
coef_df$source    <- factor(coef_df$source,
                             levels = c("Empirical",
                                        "OpenAI", "Anthropic", "Google/Gemini",
                                        "xAI/Grok", "Ollama",
                                        "Bedrock/Claude", "Bedrock/Llama"))

provider_palette <- c(
  "Empirical"      = "#2166AC",
  "OpenAI"         = "#D6604D",
  "Anthropic"      = "#4DAC26",
  "Google/Gemini"  = "#A858A0",
  "xAI/Grok"       = "#F4A736",
  "Ollama"         = "#878787",
  "Bedrock/Claude" = "#1B7837",
  "Bedrock/Llama"  = "#762A83"
)
# Keep only levels that are present in the data
active_palette <- provider_palette[levels(coef_df$source)]

# Clip x axis to the range of finite, non-extreme values so that a single
# poorly identified coefficient (e.g. perfect separation) does not compress
# all other estimates into a sliver. Points/CIs outside the range are clipped
# but not removed from the data (coord_cartesian, not xlim in scale).
plot_xlim <- local({
  finite_vals <- c(coef_df$coef, coef_df$lower95, coef_df$upper95)
  finite_vals <- finite_vals[is.finite(finite_vals)]
  q <- quantile(abs(finite_vals), 0.98, na.rm = TRUE)
  c(-q, q) * 1.05
})

ggplot(coef_df, aes(x = coef, y = term, colour = source,
                    xmin = lower95, xmax = upper95)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_linerange(position = position_dodge(width = 0.7), linewidth = 0.7) +
  geom_point(position   = position_dodge(width = 0.7), size = 2.0) +
  scale_colour_manual(values = active_palette) +
  coord_cartesian(xlim = plot_xlim) +
  facet_wrap(~ condition, nrow = 1) +
  labs(
    title  = "REM Coefficients: Empirical vs. LLM (all providers × all conditions)",
    x      = "Coefficient (log hazard ratio)",
    y      = NULL,
    colour = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position  = "bottom",
    strip.text       = element_text(size = 8),
    panel.spacing    = unit(0.8, "lines")
  )

ggsave(
  file.path(STUDY_DIR, "study3_coef_comparison.pdf"),
  width = 16, height = 6
)
cat("Combined forest plot saved.\n")

# =============================================================================
# 12. SAVE ALL RESULTS
# =============================================================================

save(rem_empirical, results, coef_df, edgelist,
     file = file.path(STUDY_DIR, "study3_results.RData"))
cat("Results saved to", file.path(STUDY_DIR, "study3_results.RData"), "\n")

# =============================================================================
# 13. SUFFICIENT STATISTIC COMPARISON
#
# Instead of comparing Cox model coefficients, compare the mean values of each
# REM covariate across realized events — the sufficient statistics. This avoids
# convergence issues that inflate or invert coefficients in short sequences, and
# gives a more direct read on whether the LLM sequence has similar structural
# properties to the empirical one.
# =============================================================================

cat("\nComputing sufficient statistic means...\n")

# Terms to compare — exclude strata dummies returned by compute_seq_stat_means
stat_terms <- c("Imitation", "Repetition", "RRecInitiation",
                "DyadRecencyLog", "IOviolation", "NTriadic")

emp_stat_means <- compute_seq_stat_means(edgelist, rem_formula, base_riskset)
emp_stat_means <- emp_stat_means[stat_terms]
cat("Empirical sufficient stat means:\n")
print(round(emp_stat_means, 3))

llm_stat_means <- lapply(names(results), function(key) {
  cat("  Computing stats:", key, "\n")
  tryCatch({
    s <- compute_seq_stat_means(results[[key]]$sequence, rem_formula, base_riskset)
    s[stat_terms]
  }, error = function(e) {
    warning("compute_seq_stat_means failed for ", key, ": ",
            conditionMessage(e), call. = FALSE)
    NULL
  })
})
names(llm_stat_means) <- names(results)
llm_stat_means <- Filter(Negate(is.null), llm_stat_means)

# Assemble long data frame for plotting
stat_term_order <- stat_terms

stat_df <- do.call(rbind, c(
  list(data.frame(
    term      = names(emp_stat_means),
    mean_val  = as.numeric(emp_stat_means),
    source    = "Empirical",
    condition = "Empirical",
    stringsAsFactors = FALSE
  )),
  lapply(names(llm_stat_means), function(key) {
    p <- split_key(key)
    data.frame(
      term      = names(llm_stat_means[[key]]),
      mean_val  = as.numeric(llm_stat_means[[key]]),
      source    = p$pname,
      condition = condition_labels[match(p$cid, condition_ids)],
      stringsAsFactors = FALSE
    )
  })
))

stat_df$term      <- factor(stat_df$term,      levels = rev(stat_term_order))
stat_df$condition <- factor(stat_df$condition,
                             levels = c("Empirical", condition_labels))
stat_df$source    <- factor(stat_df$source,
                             levels = c("Empirical", "OpenAI", "Anthropic",
                                        "Google/Gemini", "xAI/Grok", "Ollama",
                                        "Bedrock/Claude", "Bedrock/Llama"))

active_palette_stat <- provider_palette[levels(stat_df$source)]

# Separate empirical reference lines from LLM points for clarity
emp_stat_df <- stat_df[stat_df$source == "Empirical", ]
llm_stat_df <- stat_df[stat_df$source != "Empirical", ]

ggplot(llm_stat_df, aes(x = mean_val, y = term, colour = source)) +
  geom_vline(
    data    = emp_stat_df,
    mapping = aes(xintercept = mean_val),
    linetype = "dashed", colour = "grey40", linewidth = 0.6
  ) +
  geom_point(position = position_dodge(width = 0.6), size = 2.5) +
  scale_colour_manual(values = active_palette_stat[levels(llm_stat_df$source)]) +
  facet_wrap(~ condition, nrow = 1) +
  labs(
    title  = "Sufficient Statistic Means: Empirical (dashed) vs. LLM (all providers × all conditions)",
    x      = "Mean covariate value across realized events",
    y      = NULL,
    colour = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(size = 8),
    panel.spacing   = unit(0.8, "lines")
  )

ggsave(
  file.path(STUDY_DIR, "study3_suffstat_comparison.pdf"),
  width = 16, height = 6
)
cat("Sufficient statistic plot saved.\n")

# =============================================================================
# 13. RANKED-PREFERENCE QUERY INFRASTRUCTURE
#
# Instead of asking for a single prediction, the LLM returns a rank-ordered
# list of K preferred targets (sender fixed to the empirical actor).
# This lets us ask: does the LLM's rank ordering track the REM's hazard
# ordering?  Do higher-ranked choices produce sequences with REM coefficients
# closer to the empirical model?
# =============================================================================

#' Parse a ranked list of integer target IDs from an LLM response.
#'
#' Extracts valid integer IDs in the order they appear, deduplicates them,
#' and pads with a random sample if fewer than k unique valid IDs are found.
#'
#' @param text      Raw LLM response string.
#' @param valid_set Integer vector of valid target node IDs.
#' @param k         Number of ranked choices to return.
#' @return Integer vector of length k.
parse_ranked_response <- function(text, valid_set, k) {
  all_ints   <- suppressWarnings(
    as.integer(regmatches(text, gregexpr("[0-9]+", text))[[1L]])
  )
  valid_ints <- unique(all_ints[!is.na(all_ints) & all_ints %in% valid_set])

  if (length(valid_ints) < k) {
    remaining  <- setdiff(valid_set, valid_ints)
    pad        <- sample(remaining, min(k - length(valid_ints), length(remaining)))
    valid_ints <- c(valid_ints, pad)
  }

  as.integer(head(valid_ints, k))
}

#' Build a query factory that elicits K rank-ordered target preferences.
#'
#' Returns a closure with the same signature as other query factories:
#'   f(past_events, empirical_event, all_nodes) -> data.frame
#'
#' Unlike the single-event factories, the return value is a k-row data frame
#' with columns: rank (1..k), sender (integer), receiver (integer).
#'
#' @param cfg rem_cfg object.
#' @param k   Number of ranked preferences to request (default 5).
make_query_ranked_receiver <- function(cfg, k = 5L) {
  force(cfg); force(k)
  function(past_events, empirical_event, all_nodes) {
    sender        <- as.integer(empirical_event$sender)
    valid_targets <- setdiff(all_nodes, sender)

    history_str   <- llrem:::format_history(past_events, cfg)
    sender_label  <- llrem:::node_label(sender, cfg)
    target_str    <- paste(
      vapply(valid_targets,
             function(v) sprintf("%s (ID: %d)", llrem:::node_label(v, cfg), v),
             character(1L)),
      collapse = ", "
    )

    system_msg <- paste(
      cfg$context,
      sprintf("You are %s in this simulation.", sender_label)
    )
    user_msg <- paste0(
      "Recent events:\n", history_str, "\n\n",
      "As ", sender_label, ", rank your top ", k,
      " most likely targets in order (most likely first).\n",
      "Valid targets: ", target_str, "\n\n",
      "Reply with EXACTLY ", k, " lines using integer IDs only:\n",
      paste(paste0(seq_len(k), ". [ID]"), collapse = "\n")
    )

    messages <- list(
      list(role = "system", content = system_msg),
      list(role = "user",   content = user_msg)
    )

    raw <- cfg$provider$call(messages,
                             max_tokens  = k * 12L,
                             temperature = 0.7)

    receivers <- parse_ranked_response(raw, valid_targets, k)

    data.frame(rank     = seq_len(k),
               sender   = sender,
               receiver = receivers,
               stringsAsFactors = FALSE)
  }
}

# =============================================================================
# 14. RANKED COLLECTION FUNCTION
# =============================================================================

#' Generate rank-ordered LLM preference sequences.
#'
#' At each time step the query factory is called once and returns k ranked
#' choices.  All k are stored; rank-1 is used to advance the running sequence
#' (for history construction under use_empirical_history = FALSE).
#'
#' @param edgelist            Prepared edgelist.
#' @param cfg                 rem_cfg; provider must be set.
#' @param k                   Number of ranked choices per step (default 5).
#' @param n_events            Events to generate; NULL = full edgelist.
#' @param use_empirical_history Show empirical or self-generated history.
#' @param rate_limit_secs     Pause between API calls.
#' @param seed                RNG seed for padding fallbacks.
#' @param checkpoint_path     File path for incremental saves; NULL = off.
#'
#' @return Named list:
#'   $ranked_df  Long data frame: event_t, rank, sender, receiver, is_empirical
#'   $sequences  Named list of k sequence data frames ("rank1"…"rankK")
run_ranked_llm <- function(edgelist,
                            cfg,
                            k                     = 5L,
                            n_events              = NULL,
                            use_empirical_history = TRUE,
                            rate_limit_secs       = 0,
                            seed                  = 9000L,
                            checkpoint_path       = NULL) {

  all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
  n_ev         <- if (is.null(n_events)) nrow(edgelist)
                  else min(n_events, nrow(edgelist))
  emp          <- edgelist[seq_len(n_ev), ]
  query_fn     <- make_query_ranked_receiver(cfg, k)

  set.seed(seed)

  # --- Checkpoint recovery -------------------------------------------------
  ranked_rows <- list()
  start_t     <- 2L

  if (!is.null(checkpoint_path) && file.exists(checkpoint_path)) {
    env_ckpt <- new.env(parent = emptyenv())
    load(checkpoint_path, envir = env_ckpt)
    ranked_rows <- env_ckpt$ranked_rows
    start_t     <- length(ranked_rows) + 2L
    message("[llrem] Resuming ranked run from event ", start_t)
  }

  # Seed: rank-1 sequence starts with empirical event 1
  seed_row <- emp[1L, c("time", "sender", "receiver", "eveid")]
  llm_seq  <- seed_row   # only used when use_empirical_history = FALSE

  pb <- utils::txtProgressBar(min = start_t, max = n_ev, style = 3)

  for (t in start_t:n_ev) {

    history <- if (use_empirical_history) emp[seq_len(t - 1L), ] else llm_seq

    ranked <- tryCatch(
      query_fn(past_events     = history,
               empirical_event = emp[t, ],
               all_nodes       = all_nodes),
      error = function(e) {
        message("\n[llrem] ranked query failed at t=", t, ": ", conditionMessage(e))
        NULL
      }
    )

    if (is.null(ranked)) {
      # Fallback: random k targets for this step
      valid_tgts <- setdiff(all_nodes, as.integer(emp$sender[t]))
      receivers  <- sample(valid_tgts, min(k, length(valid_tgts)))
      ranked     <- data.frame(rank     = seq_along(receivers),
                               sender   = as.integer(emp$sender[t]),
                               receiver = as.integer(receivers),
                               stringsAsFactors = FALSE)
    }

    ranked$event_t      <- t
    ranked$is_empirical <- ranked$receiver == as.integer(emp$receiver[t])
    ranked_rows[[t - 1L]] <- ranked

    # Advance rank-1 sequence (for self-generated history mode)
    if (!use_empirical_history) {
      r1 <- ranked[ranked$rank == 1L, ]
      new_row <- data.frame(time     = emp$time[t],
                            sender   = r1$sender,
                            receiver = r1$receiver,
                            eveid    = paste0(r1$sender, "_", r1$receiver),
                            stringsAsFactors = FALSE)
      llm_seq <- rbind(llm_seq, new_row)
    }

    if (!is.null(checkpoint_path) && (t %% 10L == 0L || t == n_ev))
      save(ranked_rows, file = checkpoint_path)

    utils::setTxtProgressBar(pb, t)
    Sys.sleep(rate_limit_secs)
  }
  close(pb)

  # --- Assemble output -----------------------------------------------------
  ranked_df <- do.call(rbind, ranked_rows)
  row.names(ranked_df) <- NULL

  # Build one sequence per rank level
  sequences <- lapply(seq_len(k), function(r) {
    r_rows <- ranked_df[ranked_df$rank == r, ]
    seq_df <- rbind(
      seed_row,
      data.frame(time     = emp$time[r_rows$event_t],
                 sender   = r_rows$sender,
                 receiver = r_rows$receiver,
                 eveid    = paste0(r_rows$sender, "_", r_rows$receiver),
                 stringsAsFactors = FALSE)
    )
    row.names(seq_df) <- NULL
    seq_df
  })
  names(sequences) <- paste0("rank", seq_len(k))

  list(ranked_df = ranked_df, sequences = sequences)
}

# =============================================================================
# 15. RANKED STUDY  — run interactively
# Requires: providers, rem_formula, edgelist, base_riskset, STUDY_DIR
# =============================================================================

if (FALSE) {

# ── Choose provider and k ────────────────────────────────────────────────────
ranked_k   <- 5L
ranked_prov <- providers[["ollama"]]   # swap to any provider

ranked_cfg <- rem_cfg(
  provider       = ranked_prov,
  context        = paste(
    "You are participating in a simulation of an international militarized",
    "conflict network. Actors are sovereign countries. Each event represents",
    "one country using military force against another."
  ),
  action_verb    = "attacked",
  actor_noun     = "Country",
  node_labels    = node_labels,
  history_window = 30L
)

ranked_ckpt_dir <- file.path(STUDY_DIR, "checkpoints_ranked")
dir.create(ranked_ckpt_dir, showWarnings = FALSE, recursive = TRUE)

# ── Collect ranked preferences ───────────────────────────────────────────────
cat("Collecting ranked preferences (k =", ranked_k, ")...\n")

ranked_run <- run_ranked_llm(
  edgelist              = edgelist,
  cfg                   = ranked_cfg,
  k                     = ranked_k,
  n_events              = 500L,
  use_empirical_history = TRUE,
  rate_limit_secs       = rate_limits[[ranked_prov$name]],
  seed                  = 7001L,
  checkpoint_path       = file.path(ranked_ckpt_dir,
                                    paste0(ranked_prov$model, "_ranked.RData"))
)

save(ranked_run, file = file.path(STUDY_DIR, "ranked_run.RData"))
cat("Ranked sequences saved.\n")

# ── Fit one REM per rank level ───────────────────────────────────────────────
cat("Fitting REMs for each rank level...\n")

ranked_rems <- lapply(seq_len(ranked_k), function(r) {
  cat("  Fitting rank", r, "...\n")
  tryCatch(
    fit_rem(ranked_run$sequences[[r]], formula = rem_formula,
            n_nonevents = 200L, base_riskset = base_riskset),
    error = function(e) {
      warning("fit_rem failed for rank ", r, ": ", conditionMessage(e))
      NULL
    }
  )
})
names(ranked_rems) <- paste0("Rank ", seq_len(ranked_k))

# ── Recall@K and MRR ─────────────────────────────────────────────────────────
ranked_df <- ranked_run$ranked_df

# For each event, was the empirical receiver in the LLM's top-K list?
hit_by_event <- tapply(ranked_df$is_empirical, ranked_df$event_t, any)
recall_at_k  <- mean(hit_by_event, na.rm = TRUE)
cat(sprintf("\nRecall@%d: %.1f%% of events had the empirical choice in the LLM list\n",
            ranked_k, recall_at_k * 100))

# Mean Reciprocal Rank of the empirical event
mrr_by_event <- tapply(seq_len(nrow(ranked_df)),
                       ranked_df$event_t,
                       function(idx) {
                         hit_idx <- which(ranked_df$is_empirical[idx])
                         if (length(hit_idx) == 0L) 0 else 1 / min(hit_idx)
                       })
mrr <- mean(unlist(mrr_by_event), na.rm = TRUE)
cat(sprintf("Mean Reciprocal Rank (MRR): %.3f\n", mrr))

# ── Forest plot: coefficients by rank vs empirical ───────────────────────────
ranked_coef_df <- do.call(rbind, c(
  list(extract_coefs(rem_empirical, "Empirical", "Empirical")),
  lapply(seq_len(ranked_k), function(r) {
    if (is.null(ranked_rems[[r]])) return(NULL)
    extract_coefs(ranked_rems[[r]], paste0("Rank ", r), paste0("Rank ", r))
  })
))

ranked_coef_df$source <- factor(
  ranked_coef_df$source,
  levels = c("Empirical", paste0("Rank ", seq_len(ranked_k)))
)
ranked_coef_df$term <- factor(ranked_coef_df$term,
                               levels = rev(term_order))

ggplot(ranked_coef_df,
       aes(x = coef, y = term, colour = source,
           xmin = lower95, xmax = upper95)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_linerange(position = position_dodge(width = 0.7), linewidth = 0.7) +
  geom_point(position   = position_dodge(width = 0.7), size = 2) +
  scale_colour_brewer(palette = "RdYlBu", direction = -1) +
  labs(
    title  = sprintf("REM Coefficients by LLM Rank Level  (k = %d, %s)",
                     ranked_k, ranked_prov$model),
    x      = "Coefficient (log hazard ratio)",
    y      = NULL,
    colour = "Rank level"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(file.path(STUDY_DIR, "ranked_coef_by_rank.pdf"), width = 8, height = 5)
cat("Ranked coefficient plot saved.\n")

# ── Recall@k curve ────────────────────────────────────────────────────────────
recall_curve <- vapply(seq_len(ranked_k), function(kk) {
  hits <- tapply(ranked_df$is_empirical[ranked_df$rank <= kk],
                 ranked_df$event_t[ranked_df$rank <= kk],
                 any)
  mean(hits, na.rm = TRUE)
}, numeric(1L))

recall_df <- data.frame(k = seq_len(ranked_k), recall = recall_curve)

ggplot(recall_df, aes(x = k, y = recall)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_x_continuous(breaks = seq_len(ranked_k)) +
  labs(
    title = sprintf("Recall@k — %s", ranked_prov$model),
    x     = "k (top-k list size)",
    y     = "Recall (empirical event in top-k)"
  ) +
  theme_minimal(base_size = 12)

ggsave(file.path(STUDY_DIR, "ranked_recall_curve.pdf"), width = 5, height = 4)
cat("Recall curve saved.\n")

} # end if (FALSE)
