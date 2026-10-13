# =============================================================================
# llm_rem_multidomain.R
# Cross-domain LLM-REM sensitivity study
#
# Research question: Are LLM endogenous tendencies (Repetition, RRecInitiation,
# NTriadic, DyadRecencyLog) sensitive to interaction context, or are they stable
# properties of LLM behaviour regardless of social domain?
#
# For each domain we run two conditions:
#   sender_obs  — LLM roleplays the empirical sender; observed history
#   predict_obs — LLM predicts next sender AND receiver; observed history
#
# Results are compared against each domain's empirical REM baseline using
# sufficient statistic means and RMSE.
#
# Domains included:
#   War        — international conflict (high visibility, conflictual)
#   Classroom  — spoken utterances in a high-school class (high visibility, affiliative)
#   Gangs      — simulated gang violence (high visibility, conflictual)
#   Email      — internal email at a research institution (low visibility, affiliative)
#   Twitter    — public reply/mention network (high visibility, affiliative)
#   Phone      — military command-and-control phone calls (high visibility, affiliative)
#
# Domains excluded:
#   ftf        — only 7 actors; trivial target space for LLM
#   PMs / SMS  — 600+ actors; impractical to enumerate valid targets in prompt
#   finch      — non-human interaction network
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

ACTIVE_PROVIDERS <- c("bedrock_claude", "bedrock_llama")

ALL_PROVIDERS <- list(
  bedrock_claude = list(
    provider   = llm_provider_bedrock(
      model = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
      name  = "Bedrock/Claude (Haiku 4.5)"
    ),
    rate_limit = 1
  ),
  bedrock_llama = list(
    provider   = llm_provider_bedrock(
      model = "us.meta.llama3-1-70b-instruct-v1:0",
      name  = "Bedrock/Llama (3.1 70B)"
    ),
    rate_limit = 1
  )
)

providers   <- lapply(ALL_PROVIDERS[ACTIVE_PROVIDERS], `[[`, "provider")
rate_limits <- vapply(ALL_PROVIDERS[ACTIVE_PROVIDERS], `[[`, numeric(1L), "rate_limit")
names(providers)   <- ACTIVE_PROVIDERS
names(rate_limits) <- ACTIVE_PROVIDERS

# =============================================================================
# 2. DOMAIN CONFIGURATIONS
#
# context      — system-message framing for the LLM
# action_verb  — past-tense verb describing a directed event
# actor_noun   — capitalised singular noun for nodes
# data_path    — path to the .Rdata file
# n_events     — number of events to generate (NULL = full sequence)
# =============================================================================

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root
DATA_DIR  <- file.path(STUDY_DIR, "DataP")

DOMAIN_CONFIGS <- list(

  War = list(
    context     = paste(
      "You are participating in a simulation of an international militarized",
      "conflict network. Actors are sovereign countries. Each event represents",
      "one country using military force against another."
    ),
    action_verb = "attacked",
    actor_noun  = "Country",
    data_path   = file.path(DATA_DIR, "War/conflict.Rdata"),
    n_events    = 1000L
  ),

  Classroom = list(
    context     = paste(
      "You are participating in a simulation of spoken interaction in a",
      "high-school classroom. Actors are students and a teacher. Each event",
      "represents one participant directing a spoken utterance to another.",
      "Node 21 represents the class as a whole (group address)."
    ),
    action_verb = "spoke to",
    actor_noun  = "Participant",
    data_path   = file.path(DATA_DIR, "Classroom/McFarland.Rdata"),
    n_events    = NULL   # use full 691-event sequence
  ),

  Gangs = list(
    context     = paste(
      "You are participating in a simulation of gang violence in an urban",
      "neighbourhood. Actors are street gangs. Each event represents one gang",
      "carrying out a violent act against another gang."
    ),
    action_verb = "attacked",
    actor_noun  = "Gang",
    data_path   = file.path(DATA_DIR, "Gangs/gangnet1.Rdata"),
    n_events    = NULL   # use full 450-event sequence
  ),

  Email = list(
    context     = paste(
      "You are participating in a simulation of email communication within a",
      "research organisation. Actors are employees. Each event represents one",
      "employee sending an email to another employee."
    ),
    action_verb = "emailed",
    actor_noun  = "Employee",
    data_path   = file.path(DATA_DIR, "Email/eu1.txt1.Rdata"),
    n_events    = 1000L
  ),

  Twitter = list(
    context     = paste(
      "You are participating in a simulation of a Twitter conversation.",
      "Actors are users. Each event represents one user sending a public",
      "reply or mention to another user."
    ),
    action_verb = "replied to",
    actor_noun  = "User",
    data_path   = file.path(DATA_DIR, "Twitter/twitter_boston_relevent.rdata"),
    n_events    = NULL   # use full 227-event sequence
  ),

  Phone = list(
    context     = paste(
      "You are participating in a simulation of phone calls during a military",
      "command-and-control exercise. Actors are organisational roles. Each",
      "event represents one role placing a call to another role."
    ),
    action_verb = "called",
    actor_noun  = "Role",
    data_path   = file.path(DATA_DIR, "Phone/Phone.Rdata"),
    n_events    = NULL   # use full 196-event sequence (REM may be unstable)
  )

)

ACTIVE_DOMAINS <- c("War", "Classroom", "Gangs", "Email", "Twitter", "Phone")

# Conditions to run for each domain × provider combination
CONDITIONS <- list(
  list(id = "sender_obs",  label = "Sender roleplay"),
  list(id = "predict_obs", label = "Predict")
)
condition_ids    <- vapply(CONDITIONS, `[[`, character(1L), "id")
condition_labels <- setNames(
  vapply(CONDITIONS, `[[`, character(1L), "label"), condition_ids
)

# =============================================================================
# 3. LOADER
#
# Standardises any DataP edgelist file into the format expected by llrem:
# a data frame with columns (time, sender, receiver, eveid), all integer.
# Handles: named data frames, unnamed matrices, variant column names.
# =============================================================================

load_datap <- function(path) {
  e  <- new.env(parent = emptyenv())
  load(path, envir = e)
  el <- as.data.frame(e$edgelist)
  colnames(el)[1:3] <- c("time", "sender", "receiver")
  el$time     <- as.integer(el$time)
  el$sender   <- as.integer(el$sender)
  el$receiver <- as.integer(el$receiver)
  el <- el[el$sender != el$receiver, ]   # remove any self-loops
  el$eveid <- paste0(el$sender, "_", el$receiver)
  rownames(el) <- NULL
  list(edgelist = el, n = as.integer(e$n))
}

# =============================================================================
# 4. REM FORMULA AND SUFFICIENT STATISTIC TERMS
# =============================================================================

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

stat_terms <- c("Imitation", "Repetition", "RRecInitiation",
                "DyadRecencyLog", "IOviolation", "NTriadic")

# =============================================================================
# 5. HELPERS
# =============================================================================

split_key <- function(key, active_providers, active_domains) {
  pkey <- active_providers[startsWith(key, paste0(active_providers, "_"))]
  if (length(pkey) != 1L) stop("Cannot resolve provider for key: ", key)
  rest   <- substring(key, nchar(pkey) + 2L)   # "Domain_condition_id"
  domain <- active_domains[startsWith(rest, paste0(active_domains, "_"))]
  if (length(domain) != 1L) stop("Cannot resolve domain for key: ", key)
  cid    <- substring(rest, nchar(domain) + 2L)
  list(pkey = pkey, domain = domain, cid = cid)
}

make_query_fn <- function(condition_id, cfg) {
  switch(condition_id,
    sender_obs  = make_query_roleplay_sender(cfg),
    predict_obs = make_query_predict(cfg),
    stop("Unknown condition: ", condition_id)
  )
}

# =============================================================================
# 6. MAIN LOOP
# =============================================================================

results  <- list()
emp_rems <- list()
emp_sms  <- list()

for (domain_id in ACTIVE_DOMAINS) {
  dcfg <- DOMAIN_CONFIGS[[domain_id]]
  cat(sprintf("\n%s\n%s\n", strrep("=", 70), domain_id))

  # --- Load data ---
  dat          <- load_datap(dcfg$data_path)
  edgelist     <- dat$edgelist
  all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
  base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)
  n_ev         <- if (is.null(dcfg$n_events)) nrow(edgelist) else
                    min(dcfg$n_events, nrow(edgelist))

  cat(sprintf("  Events: %d (using %d)  |  Actors: %d\n",
              nrow(edgelist), n_ev, length(all_nodes)))

  # --- Empirical REM ---
  cat("  Fitting empirical REM...\n")
  rem_emp <- tryCatch(
    fit_rem(edgelist[seq_len(n_ev), ], rem_formula,
            n = length(all_nodes), n_nonevents = 200L,
            base_riskset = base_riskset),
    error = function(e) {
      message("  [WARN] Empirical REM failed: ", e$message); NULL
    }
  )
  emp_rems[[domain_id]] <- rem_emp
  if (!is.null(rem_emp))
    cat("  Empirical coefs:", paste(round(coef(rem_emp), 2), collapse = "  "), "\n")

  # --- Empirical sufficient statistics ---
  emp_sm <- tryCatch(
    compute_seq_stat_means(edgelist[seq_len(n_ev), ],
                            rem_formula, base_riskset)[stat_terms],
    error = function(e) { message("  [WARN] emp suffstats failed"); NULL }
  )
  emp_sms[[domain_id]] <- emp_sm

  # --- LLM conditions ---
  for (provider_id in ACTIVE_PROVIDERS) {
    provider <- providers[[provider_id]]
    rl       <- rate_limits[[provider_id]]

    cfg <- rem_cfg(
      provider       = provider,
      context        = dcfg$context,
      action_verb    = dcfg$action_verb,
      actor_noun     = dcfg$actor_noun,
      node_labels    = NULL,
      history_window = 30L
    )

    for (cond in CONDITIONS) {
      key  <- paste(provider_id, domain_id, cond$id, sep = "_")
      ckpt <- file.path(STUDY_DIR,
                        sprintf("checkpoints_multidomain/%s.RData", key))
      dir.create(dirname(ckpt), showWarnings = FALSE, recursive = TRUE)

      # Skip if already complete
      if (file.exists(ckpt)) {
        env <- new.env(parent = emptyenv())
        load(ckpt, envir = env)
        n_completed <- length(env$llm_rows)
        if (n_completed >= n_ev - 1L) {
          cat(sprintf("  [SKIP] %s already complete (%d events)\n",
                      key, n_completed))
          # Reconstruct result from checkpoint for results list
          if (!key %in% names(results)) {
            seq_df <- rbind(
              edgelist[1L, c("time", "sender", "receiver", "eveid")],
              do.call(rbind, env$llm_rows)
            )
            rem_fit <- tryCatch(
              fit_rem(seq_df, rem_formula,
                      n = length(all_nodes), n_nonevents = 200L,
                      base_riskset = base_riskset),
              error = function(e) NULL
            )
            results[[key]] <- list(sequence = seq_df, rem = rem_fit)
          }
          next
        }
      }

      cat(sprintf("\n  Running: %s | %s | %s\n",
                  domain_id, provider$name, cond$label))
      query_fn <- make_query_fn(cond$id, cfg)

      res <- tryCatch(
        run_llm_rem(
          edgelist              = edgelist,
          formula               = rem_formula,
          query_fn              = query_fn,
          n_events              = n_ev,
          n                     = length(all_nodes),
          n_nonevents           = 200L,
          base_riskset          = base_riskset,
          seed                  = 8214L,
          checkpoint_path       = ckpt,
          use_empirical_history = TRUE,
          rate_limit_secs       = rl,
          fit                   = (n_ev >= 100L)
        ),
        error = function(e) {
          message("  [ERROR] ", key, ": ", e$message); NULL
        }
      )
      results[[key]] <- res
      cat(sprintf("  Done: %s\n", key))
    }
  }
}

# =============================================================================
# 7. CROSS-DOMAIN SUFFICIENT STATISTIC COMPARISON
# =============================================================================

cat("\n\n=== Cross-domain sufficient statistic comparison ===\n\n")

# Build a long data frame: domain × source × covariate → mean_val
rows <- list()

for (domain_id in ACTIVE_DOMAINS) {
  sm <- emp_sms[[domain_id]]
  if (!is.null(sm))
    rows[[length(rows) + 1L]] <- data.frame(
      domain    = domain_id,
      source    = "Empirical",
      condition = "Empirical",
      term      = names(sm),
      mean_val  = as.numeric(sm),
      stringsAsFactors = FALSE
    )

  for (key in grep(paste0("_", domain_id, "_"), names(results), value = TRUE)) {
    res <- results[[key]]
    if (is.null(res)) next
    k <- split_key(key, ACTIVE_PROVIDERS, ACTIVE_DOMAINS)
    sm_llm <- tryCatch(
      compute_seq_stat_means(res$sequence, rem_formula,
                              llrem:::get_base_riskset(
                                NULL,
                                sort(unique(c(res$sequence$sender,
                                              res$sequence$receiver))))
      )[stat_terms],
      error = function(e) NULL
    )
    if (is.null(sm_llm)) next
    rows[[length(rows) + 1L]] <- data.frame(
      domain    = domain_id,
      source    = providers[[k$pkey]]$name,
      condition = condition_labels[[k$cid]],
      term      = names(sm_llm),
      mean_val  = as.numeric(sm_llm),
      stringsAsFactors = FALSE
    )
  }
}

stat_df <- do.call(rbind, rows)
stat_df$term      <- factor(stat_df$term, levels = rev(stat_terms))
stat_df$domain    <- factor(stat_df$domain, levels = ACTIVE_DOMAINS)
stat_df$condition <- factor(stat_df$condition,
                             levels = c("Empirical", condition_labels))

# RMSE table
cat("RMSE by domain and condition:\n")
rmse_rows <- list()
for (domain_id in ACTIVE_DOMAINS) {
  sm_emp <- emp_sms[[domain_id]]
  if (is.null(sm_emp)) next
  for (key in grep(paste0("_", domain_id, "_"), names(results), value = TRUE)) {
    res <- results[[key]]
    if (is.null(res)) next
    k <- split_key(key, ACTIVE_PROVIDERS, ACTIVE_DOMAINS)
    sm_llm <- tryCatch(
      compute_seq_stat_means(
        res$sequence, rem_formula,
        llrem:::get_base_riskset(
          NULL,
          sort(unique(c(res$sequence$sender, res$sequence$receiver))))
      )[stat_terms],
      error = function(e) NULL
    )
    if (is.null(sm_llm)) next
    rmse <- sqrt(mean((sm_llm - sm_emp)^2))
    rmse_rows[[length(rmse_rows) + 1L]] <- data.frame(
      Domain    = domain_id,
      Provider  = providers[[k$pkey]]$name,
      Condition = condition_labels[[k$cid]],
      RMSE      = round(rmse, 3),
      stringsAsFactors = FALSE
    )
  }
}
rmse_df <- do.call(rbind, rmse_rows)
rmse_df <- rmse_df[order(rmse_df$Domain, rmse_df$RMSE), ]
print(rmse_df, row.names = FALSE)

# =============================================================================
# 8. PLOTS
# =============================================================================

# --- Per-domain faceted dot plot (one facet per stat, one panel per domain) ---
emp_s <- stat_df[stat_df$source == "Empirical", ]
llm_s <- stat_df[stat_df$source != "Empirical", ]

provider_palette <- c(
  "Empirical"                  = "#2166AC",
  "Bedrock/Claude (Haiku 4.5)" = "#1B7837",
  "Bedrock/Llama (3.1 70B)"    = "#762A83"
)

p_suffstat <- ggplot(mapping = aes(x = mean_val, y = domain)) +
  geom_segment(data = llm_s,
               aes(x = -Inf, xend = mean_val, y = domain, yend = domain,
                   group = interaction(domain, condition)),
               colour = "grey72", linewidth = 0.35) +
  geom_segment(data = emp_s,
               aes(x = -Inf, xend = mean_val, y = domain, yend = domain),
               colour = "#2166AC", linewidth = 0.35, alpha = 0.5) +
  geom_point(data = llm_s,
             aes(colour = source, shape = condition),
             size = 3, alpha = 0.6) +
  geom_point(data = emp_s,
             aes(colour = source),
             shape = 18, size = 3.5, alpha = 0.85) +
  scale_colour_manual(values = provider_palette) +
  scale_shape_manual(values = c("Sender roleplay" = 16, "Predict" = 17)) +
  facet_wrap(~ term, scales = "free_x", nrow = 2) +
  labs(x = "Mean covariate value across realized events",
       y = NULL, colour = NULL, shape = "Condition",
       title   = "Sufficient statistic means by domain",
       caption = "Diamond = empirical baseline.") +
  theme_minimal(base_size = 11) +
  theme(legend.position   = "bottom",
        strip.text         = element_text(size = 9, face = "bold"),
        strip.background   = element_rect(fill = "grey92", colour = "grey60", linewidth = 0.5),
        panel.border       = element_rect(colour = "grey60", fill = NA, linewidth = 0.5),
        panel.grid.major.y = element_blank(),
        panel.grid.minor   = element_blank(),
        panel.spacing      = unit(1, "lines"))

print(p_suffstat)
ggsave(file.path(STUDY_DIR, "multidomain_suffstat.pdf"),
       p_suffstat, width = 13, height = 7)

# --- NTriadic focus plot: each domain as a row, empirical vs LLM conditions ---
ntriadic_df <- stat_df[stat_df$term == "NTriadic", ]

p_ntriadic <- ggplot(mapping = aes(x = mean_val, y = domain, colour = source)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_point(data = ntriadic_df[ntriadic_df$source != "Empirical", ],
             aes(shape = condition),
             position = position_dodge(width = 0.6), size = 4) +
  geom_point(data = ntriadic_df[ntriadic_df$source == "Empirical", ],
             shape = 18, size = 4.5) +
  scale_colour_manual(values = provider_palette) +
  scale_shape_manual(values = c("Sender roleplay" = 16, "Predict" = 17)) +
  labs(x = "NTriadic mean", y = NULL, colour = NULL, shape = "Condition",
       title = "NTriadic sufficient statistic by domain",
       subtitle = "Negative = empirically correct; positive = LLM 'enemy of my enemy' bias") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

print(p_ntriadic)
ggsave(file.path(STUDY_DIR, "multidomain_ntriadic.pdf"),
       p_ntriadic, width = 8, height = 5)

# =============================================================================
# 9. SAVE
# =============================================================================

save(results, emp_rems, emp_sms, stat_df, rmse_df,
     file = file.path(STUDY_DIR, "multidomain_results.RData"))
cat(sprintf("\nResults saved to %s\n",
            file.path(STUDY_DIR, "multidomain_results.RData")))
