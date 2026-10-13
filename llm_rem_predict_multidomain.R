# =============================================================================
# llm_rem_predict_multidomain.R
# Multi-domain LLM prediction study (predict_obs condition)
#
# Research question: How well does an LLM predict relational events across
# social domains when asked to name both sender and receiver from observed
# history, with no structural scaffolding?
#
# This is the baseline condition for comparison with the multi-agent study
# (llm_rem_multiagent_multidomain.R), where sender activation is governed by
# the empirical REM hazard surface.
#
# Design:
#   run_llm_rem() with make_query_predict() and use_empirical_history = TRUE.
#   The LLM sees the empirical event history at each step and predicts the
#   next (sender, receiver) pair. No REM coefficients are used during
#   generation; the empirical REM is fitted only for the baseline comparison.
#
# Nine REM terms:
#   SndInitiation, IOviolation, Repetition, RecRecencyRank,
#   NOTPSnd, OutDegSnd, WeightedRec, Imitation, NITPSnd
#
# Domains: same 27 network cells as the multi-agent study.
# =============================================================================

library(survival)
library(ggplot2)
library(llrem)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

N_EVENTS  <- 1000L    # max events per network

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root
DATA_DIR  <- file.path(STUDY_DIR, "DataP")
CKPT_DIR  <- file.path(STUDY_DIR, "checkpoints_predict_multidomain")
dir.create(CKPT_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 2. CREDENTIALS (AWS Bedrock via environment variables)
#    Set these in your shell or .Renviron -- do not hardcode here.
# =============================================================================

if (nzchar(Sys.getenv("AWS_ACCESS_KEY_ID"))) {
  message("Using AWS credentials from environment.")
} else {
  stop("AWS credentials not found. Set AWS_ACCESS_KEY_ID and ",
       "AWS_SECRET_ACCESS_KEY in your environment or .Renviron.", call. = FALSE)
}

# =============================================================================
# 3. PROVIDERS
# =============================================================================

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
# 4. DOMAIN CONFIGURATIONS  (identical to multiagent study)
# =============================================================================

DOMAIN_CONFIGS <- list(

  War = list(
    data_paths  = file.path(DATA_DIR, "War/conflict.Rdata"),
    context     = paste(
      "You are participating in a simulation of an international militarized",
      "conflict network. Actors are sovereign countries. Each event represents",
      "one country using military force against another."
    ),
    action_verb = "attacked",
    actor_noun  = "Country",
    n_events    = N_EVENTS,
    valence     = "conflictual",
    visibility  = "high"
  ),

  Classroom = list(
    data_paths  = file.path(DATA_DIR, "Classroom/McFarland.Rdata"),
    context     = paste(
      "You are participating in a simulation of spoken interaction in a",
      "high-school classroom. Actors are students and a teacher. Each event",
      "represents one participant directing a spoken utterance to another.",
      "Node 21 represents the class as a whole (group address)."
    ),
    action_verb = "spoke to",
    actor_noun  = "Participant",
    n_events    = NULL,
    valence     = "affiliative",
    visibility  = "high"
  ),

  Gangs = list(
    data_paths  = file.path(DATA_DIR, "Gangs",
                            paste0("gangnet", 1:5, ".Rdata")),
    context     = paste(
      "You are participating in a simulation of gang violence in an urban",
      "neighbourhood. Actors are street gangs. Each event represents one gang",
      "carrying out a violent act against another gang."
    ),
    action_verb = "attacked",
    actor_noun  = "Gang",
    n_events    = NULL,
    valence     = "conflictual",
    visibility  = "high"
  ),

  Email = list(
    data_paths  = file.path(DATA_DIR, "Email",
                            c("eu1.txt1.Rdata", "eu2.txt1.Rdata",
                              "eu3.txt1.Rdata", "eu4.txt1.Rdata",
                              "eu1.txt5001.Rdata")),
    context     = paste(
      "You are participating in a simulation of email communication within a",
      "research organisation. Actors are employees. Each event represents one",
      "employee sending an email to another employee."
    ),
    action_verb = "emailed",
    actor_noun  = "Employee",
    n_events    = N_EVENTS,
    valence     = "affiliative",
    visibility  = "low"
  ),

  ftf = list(
    data_paths  = file.path(DATA_DIR, "ftf",
                            paste0("network", 0:4, ".Rdata")),
    context     = paste(
      "You are participating in a simulation of face-to-face interaction",
      "during a group task. Actors are participants in a board game. Each event",
      "represents one participant directing their visual attention (a glance)",
      "to another, often in secret."
    ),
    action_verb = "looked at",
    actor_noun  = "Participant",
    n_events    = N_EVENTS,
    valence     = "affiliative",
    visibility  = "low"
  ),

  PMs = list(
    data_paths  = file.path(DATA_DIR, "PMs",
                            c("ucimsg1.Rdata", "ucimsg7001.Rdata",
                              "ucimsg14001.Rdata", "ucimsg21001.Rdata",
                              "ucimsg28001.Rdata")),
    context     = paste(
      "You are participating in a simulation of private messaging on an",
      "online college social platform. Actors are students. Each event",
      "represents one student sending a private message to another."
    ),
    action_verb = "messaged",
    actor_noun  = "Student",
    n_events    = N_EVENTS,
    valence     = "affiliative",
    visibility  = "low"
  ),

  SMS = list(
    data_paths  = file.path(DATA_DIR, "SMS",
                            c("sms5001.Rdata", "sms10001.Rdata",
                              "sms15001.Rdata")),
    context     = paste(
      "You are participating in a simulation of SMS text messaging.",
      "Actors are mobile phone users. Each event represents one user",
      "sending a text message to another user."
    ),
    action_verb = "texted",
    actor_noun  = "User",
    n_events    = N_EVENTS,
    valence     = "affiliative",
    visibility  = "low"
  ),

  Twitter = list(
    data_paths  = file.path(DATA_DIR, "Twitter/twitter_boston_relevent.rdata"),
    context     = paste(
      "You are participating in a simulation of a Twitter conversation.",
      "Actors are users. Each event represents one user sending a public",
      "reply or mention to another user."
    ),
    action_verb = "replied to",
    actor_noun  = "User",
    n_events    = NULL,
    valence     = "affiliative",
    visibility  = "high"
  ),

  Phone = list(
    data_paths  = file.path(DATA_DIR, "Phone/Phone.Rdata"),
    context     = paste(
      "You are participating in a simulation of phone calls during a military",
      "command-and-control exercise. Actors are organisational roles. Each",
      "event represents one role placing a call to another role."
    ),
    action_verb = "called",
    actor_noun  = "Role",
    n_events    = NULL,
    valence     = "affiliative",
    visibility  = "high"
  )

)

ACTIVE_DOMAINS <- c("War", "Classroom", "Gangs", "Email", "ftf",
                    "PMs", "SMS", "Twitter", "Phone")

# =============================================================================
# 5. FORMULA AND STAT TERMS
# =============================================================================

rem_formula <- Surv(time = time1, event = occurred) ~
  SndInitiation + IOviolation + Repetition + RecRecencyRank +
  NOTPSnd + OutDegSnd + WeightedRec + Imitation + NITPSnd +
  strata(time2)

stat_terms <- c("SndInitiation", "IOviolation", "Repetition", "RecRecencyRank",
                "NOTPSnd", "OutDegSnd", "WeightedRec", "Imitation", "NITPSnd")

# =============================================================================
# 6. LOADER
# =============================================================================

load_datap <- function(path) {
  e <- new.env(parent = emptyenv())
  load(path, envir = e)
  el <- as.data.frame(e$edgelist)
  colnames(el)[1:3] <- c("time", "sender", "receiver")
  el$time     <- as.integer(el$time)
  el$sender   <- as.integer(el$sender)
  el$receiver <- as.integer(el$receiver)
  el <- el[el$sender != el$receiver, ]
  el$eveid <- paste0(el$sender, "_", el$receiver)
  rownames(el) <- NULL
  list(edgelist = el, n = as.integer(e$n))
}

# =============================================================================
# 7. MAIN LOOP
# =============================================================================

results  <- list()   # key: provider_net_key  → run_llm_rem output
emp_rems <- list()   # key: net_key           → coxph fit
emp_sms  <- list()   # key: net_key           → named numeric

for (domain_id in ACTIVE_DOMAINS) {
  dcfg       <- DOMAIN_CONFIGS[[domain_id]]
  data_paths <- as.character(dcfg$data_paths)
  n_nets     <- length(data_paths)

  cat(sprintf("\n%s\n%s  (%d network(s))\n", strrep("=", 70), domain_id, n_nets))

  for (net_idx in seq_len(n_nets)) {
    net_key <- sprintf("%s_%02d", domain_id, net_idx)
    cat(sprintf("\n  --- %s ---\n", net_key))

    dat          <- load_datap(data_paths[net_idx])
    edgelist     <- dat$edgelist
    all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
    base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)
    n_ev         <- if (is.null(dcfg$n_events)) nrow(edgelist) else
                      min(dcfg$n_events, nrow(edgelist))

    cat(sprintf("  Actors: %d  |  Events total: %d  |  Using: %d\n",
                length(all_nodes), nrow(edgelist), n_ev))

    # --- Empirical REM ---
    cat("  Fitting empirical REM...\n")
    rem_emp <- tryCatch(
      fit_rem(edgelist[seq_len(n_ev), ], rem_formula,
              n            = length(all_nodes),
              n_nonevents  = 200L,
              base_riskset = base_riskset),
      error = function(e) {
        message("  [WARN] Empirical REM failed: ", e$message); NULL
      }
    )
    emp_rems[[net_key]] <- rem_emp
    if (!is.null(rem_emp))
      cat("  Empirical coefs:", paste(round(coef(rem_emp), 2), collapse = "  "), "\n")

    # --- Empirical sufficient statistic means ---
    emp_sm <- tryCatch(
      compute_seq_stat_means(edgelist[seq_len(n_ev), ],
                              rem_formula, base_riskset)[stat_terms],
      error = function(e) { message("  [WARN] emp suffstats failed"); NULL }
    )
    emp_sms[[net_key]] <- emp_sm

    # --- predict_obs condition per provider ---
    for (provider_id in ACTIVE_PROVIDERS) {
      run_key <- sprintf("%s_%s", provider_id, net_key)
      ckpt    <- file.path(CKPT_DIR, paste0(run_key, ".RData"))

      # Check for a complete prior run
      if (file.exists(ckpt)) {
        env <- new.env(parent = emptyenv())
        load(ckpt, envir = env)
        if (length(env$llm_rows) >= n_ev - 1L) {
          cat(sprintf("  [SKIP] %s already complete (%d events)\n",
                      run_key, length(env$llm_rows)))
          if (!run_key %in% names(results)) {
            seq_df <- rbind(
              edgelist[1L, c("time", "sender", "receiver", "eveid")],
              do.call(rbind, env$llm_rows)
            )
            results[[run_key]] <- list(sequence = seq_df)
          }
          next
        }
      }

      cat(sprintf("  Running: %s | %s\n", run_key, providers[[provider_id]]$name))

      cfg <- rem_cfg(
        provider       = providers[[provider_id]],
        context        = dcfg$context,
        action_verb    = dcfg$action_verb,
        actor_noun     = dcfg$actor_noun,
        node_labels    = NULL,
        history_window = 30L
      )

      res <- tryCatch(
        run_llm_rem(
          edgelist              = edgelist[seq_len(n_ev), ],
          formula               = rem_formula,
          query_fn              = make_query_predict(cfg),
          n_events              = n_ev,
          n                     = length(all_nodes),
          n_nonevents           = 200L,
          base_riskset          = base_riskset,
          seed                  = 8214L,
          checkpoint_path       = ckpt,
          use_empirical_history = TRUE,
          rate_limit_secs       = rate_limits[[provider_id]],
          fit                   = FALSE
        ),
        error = function(e) {
          message("  [ERROR] ", run_key, ": ", e$message); NULL
        }
      )
      results[[run_key]] <- res
      if (!is.null(res))
        cat(sprintf("  Done: %s  (events generated: %d)\n",
                    run_key, nrow(res$sequence)))
    }
  }
}

# =============================================================================
# 8. ANALYSIS
# =============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("Computing sufficient statistic means...\n\n")

stat_rows <- list()

for (domain_id in ACTIVE_DOMAINS) {
  dcfg       <- DOMAIN_CONFIGS[[domain_id]]
  data_paths <- as.character(dcfg$data_paths)

  for (net_idx in seq_along(data_paths)) {
    net_key <- sprintf("%s_%02d", domain_id, net_idx)

    sm_emp <- emp_sms[[net_key]]
    if (!is.null(sm_emp)) {
      stat_rows[[length(stat_rows) + 1L]] <- data.frame(
        domain     = domain_id,
        net_key    = net_key,
        net_idx    = net_idx,
        source     = "Empirical",
        provider   = "Empirical",
        term       = names(sm_emp),
        mean_val   = as.numeric(sm_emp),
        valence    = dcfg$valence,
        visibility = dcfg$visibility,
        stringsAsFactors = FALSE
      )
    }

    for (provider_id in ACTIVE_PROVIDERS) {
      run_key <- sprintf("%s_%s", provider_id, net_key)
      res     <- results[[run_key]]
      if (is.null(res)) next

      seq_nodes   <- sort(unique(c(res$sequence$sender, res$sequence$receiver)))
      seq_riskset <- llrem:::get_base_riskset(NULL, seq_nodes)

      sm_llm <- tryCatch(
        compute_seq_stat_means(res$sequence, rem_formula,
                                seq_riskset)[stat_terms],
        error = function(e) { message("  [WARN] LLM suffstats failed: ", e$message); NULL }
      )
      if (is.null(sm_llm)) next

      stat_rows[[length(stat_rows) + 1L]] <- data.frame(
        domain     = domain_id,
        net_key    = net_key,
        net_idx    = net_idx,
        source     = providers[[provider_id]]$name,
        provider   = provider_id,
        term       = names(sm_llm),
        mean_val   = as.numeric(sm_llm),
        valence    = dcfg$valence,
        visibility = dcfg$visibility,
        stringsAsFactors = FALSE
      )
    }
  }
}

stat_df <- do.call(rbind, stat_rows)
stat_df$term       <- factor(stat_df$term,   levels = stat_terms)
stat_df$domain     <- factor(stat_df$domain, levels = ACTIVE_DOMAINS)
stat_df$source     <- factor(stat_df$source,
                              levels = c("Empirical",
                                         vapply(providers, `[[`, character(1L), "name")))

domain_means <- aggregate(mean_val ~ domain + source + provider + term + valence + visibility,
                           data = stat_df, FUN = mean, na.rm = TRUE)

# bias_df
emp_domain <- domain_means[domain_means$source == "Empirical",
                            c("domain", "term", "mean_val")]
names(emp_domain)[names(emp_domain) == "mean_val"] <- "emp_val"
llm_domain <- domain_means[domain_means$source != "Empirical", ]
bias_df        <- merge(llm_domain, emp_domain, by = c("domain", "term"))
bias_df$bias   <- bias_df$mean_val - bias_df$emp_val

# rmse_df
rmse_rows <- list()
for (domain_id in ACTIVE_DOMAINS) {
  dcfg       <- DOMAIN_CONFIGS[[domain_id]]
  data_paths <- as.character(dcfg$data_paths)
  for (net_idx in seq_along(data_paths)) {
    net_key <- sprintf("%s_%02d", domain_id, net_idx)
    sm_emp  <- emp_sms[[net_key]]
    if (is.null(sm_emp)) next
    for (provider_id in ACTIVE_PROVIDERS) {
      run_key <- sprintf("%s_%s", provider_id, net_key)
      res     <- results[[run_key]]
      if (is.null(res)) next
      seq_nodes   <- sort(unique(c(res$sequence$sender, res$sequence$receiver)))
      seq_riskset <- llrem:::get_base_riskset(NULL, seq_nodes)
      sm_llm <- tryCatch(
        compute_seq_stat_means(res$sequence, rem_formula, seq_riskset)[stat_terms],
        error = function(e) NULL
      )
      if (is.null(sm_llm)) next
      rmse <- sqrt(mean((sm_llm - sm_emp)^2, na.rm = TRUE))
      rmse_rows[[length(rmse_rows) + 1L]] <- data.frame(
        domain  = domain_id, net_key = net_key, net_idx = net_idx,
        provider = provider_id, source = providers[[provider_id]]$name,
        RMSE = round(rmse, 3), stringsAsFactors = FALSE
      )
    }
  }
}

rmse_df     <- do.call(rbind, rmse_rows)
rmse_df     <- rmse_df[order(rmse_df$domain, rmse_df$RMSE), ]
rmse_domain <- aggregate(RMSE ~ domain + provider + source,
                          data = rmse_df, FUN = mean, na.rm = TRUE)
rmse_domain$RMSE <- round(rmse_domain$RMSE, 3)

cat("RMSE by domain and provider (averaged across networks):\n")
print(rmse_domain[order(rmse_domain$domain, rmse_domain$RMSE), ], row.names = FALSE)

# =============================================================================
# 9. SAVE
# =============================================================================

save(results, emp_rems, emp_sms, stat_df, domain_means, bias_df, rmse_df, rmse_domain,
     file = file.path(STUDY_DIR, "predict_multidomain_results.RData"))

cat(sprintf("\nResults saved to %s\n",
            file.path(STUDY_DIR, "predict_multidomain_results.RData")))
