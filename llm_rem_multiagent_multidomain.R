# =============================================================================
# llm_rem_multiagent_multidomain.R
# Multi-agent, multi-domain LLM-REM systematic bias study
#
# Research question: Do LLMs exhibit systematic endogenous biases that are
# stable across social domains, or do they interact with context (valence,
# visibility)?
#
# Design:
#   For each domain × network × provider cell, run_multiagent_rem() is used:
#   the empirical REM hazard surface governs sender activation; individual LLM
#   agents choose receivers. Sufficient statistic means of the generated
#   sequences are compared against the empirical baseline.
#
# Nine REM terms:
#   SndInitiation, IOviolation, Repetition, RecRecencyRank,
#   NOTPSnd, OutDegSnd, WeightedRec, Imitation, NITPSnd
#
# Domains (all non-finch, non-human):
#   War (1), Classroom (1), Gangs (5), Email (5), ftf (5),
#   PMs (5), SMS (3), Twitter (1), Phone (1)
#   → 27 domain × network cells × 2 providers = 54 runs
#
# Notes:
#   - All nodes in each network are used; no actor-pool cap.
#   - ftf has 7–8 actors (small target space); included as a boundary case.
#   - n_events is capped at N_EVENTS for domains with long sequences.
# =============================================================================

library(survival)
library(ggplot2)
library(llrem)
# =============================================================================
# 1. CONFIGURATION
# =============================================================================

N_TICKS  <- 1000L    # simulation ticks per run
N_EVENTS <- 1000L    # max events used for empirical REM fit + stat comparison

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root
DATA_DIR  <- file.path(STUDY_DIR, "DataP")
CKPT_DIR  <- file.path(STUDY_DIR, "checkpoints_multiagent_multidomain")
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
# 4. DOMAIN CONFIGURATIONS
#
# data_paths  — character vector; length > 1 for multi-network domains.
#               Domains with fewer than 5 networks use all available.
# context     — system-message framing for the LLM
# action_verb — past-tense verb describing a directed event
# actor_noun  — capitalised singular noun for nodes
# n_events    — max events used (NULL = full sequence)
# valence     — "affiliative" or "conflictual"  (for output annotation)
# visibility  — "high" or "low"                 (for output annotation)
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
      "during a group task. Actors are participants in a board game. Each event represents",
      "one participant directing their visual attention (a glance) to another, often in secret."
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
#
# Standardises any DataP edgelist into (time, sender, receiver, eveid).
# All nodes in the file are retained; no actor-pool filtering.
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

results  <- list()   # key: provider_domain_netidx  → run_multiagent_rem output
emp_rems <- list()   # key: domain_netidx            → coxph fit
emp_sms  <- list()   # key: domain_netidx            → named numeric

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

    if (is.null(rem_emp)) {
      cat("  [SKIP] Empirical REM failed — skipping LLM runs for", net_key, "\n")
      next
    }

    # --- LLM multi-agent run per provider ---
    for (provider_id in ACTIVE_PROVIDERS) {
      run_key <- sprintf("%s_%s", provider_id, net_key)
      ckpt    <- file.path(CKPT_DIR, paste0(run_key, ".RData"))

      # Check for a complete prior run and reconstruct if found
      if (file.exists(ckpt)) {
        env <- new.env(parent = emptyenv())
        load(ckpt, envir = env)
        if (!is.null(env$next_tick) && env$next_tick > N_TICKS) {
          cat(sprintf("  [SKIP] %s already complete\n", run_key))
          if (!run_key %in% names(results)) {
            seq_df <- do.call(rbind, env$sequence_rows)
            seq_df <- rbind(
              edgelist[1L, c("time", "sender", "receiver", "eveid")],
              seq_df
            )
            sender_probs <- do.call(rbind, lapply(env$sender_probs_list, function(p) {
              if (is.null(p)) rep(NA_real_, length(all_nodes)) else p
            }))
            colnames(sender_probs) <- as.character(all_nodes)
            results[[run_key]] <- list(
              sequence          = seq_df,
              sender_probs      = sender_probs,
              n_events_per_tick = env$n_events_per_tick
            )
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

      sim <- tryCatch(
        run_multiagent_rem(
          edgelist        = edgelist[seq_len(n_ev), ],
          rem_fit         = rem_emp,
          cfg             = cfg,
          n_ticks         = N_TICKS,
          lambda          = NULL,
          base_riskset    = base_riskset,
          cold_start      = FALSE,
          seed            = 8214L,
          checkpoint_path = ckpt
        ),
        error = function(e) {
          message("  [ERROR] ", run_key, ": ", e$message); NULL
        }
      )
      results[[run_key]] <- sim
      if (!is.null(sim))
        cat(sprintf("  Done: %s  (events generated: %d)\n",
                    run_key, nrow(sim$sequence)))
    }
  }
}

# =============================================================================
# 8. ANALYSIS
#
# For each domain × network × source, compute sufficient statistic means.
# Aggregate to domain level by averaging across networks.
# Build:
#   stat_df  — long: domain × net_key × source × term → mean_val
#   bias_df  — signed deviation (LLM - empirical) per term × domain × provider
#   rmse_df  — RMSE per net_key × provider, plus domain-level average
# =============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("Computing sufficient statistic means...\n\n")

# Build stat_df at the network level
stat_rows <- list()

for (domain_id in ACTIVE_DOMAINS) {
  dcfg       <- DOMAIN_CONFIGS[[domain_id]]
  data_paths <- as.character(dcfg$data_paths)
  n_nets     <- length(data_paths)

  for (net_idx in seq_len(n_nets)) {
    net_key <- sprintf("%s_%02d", domain_id, net_idx)

    # Empirical row
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

    # LLM rows
    for (provider_id in ACTIVE_PROVIDERS) {
      run_key <- sprintf("%s_%s", provider_id, net_key)
      sim     <- results[[run_key]]
      if (is.null(sim)) next

      # Rebuild base_riskset from the sequence's own actor set
      seq_nodes    <- sort(unique(c(sim$sequence$sender, sim$sequence$receiver)))
      seq_riskset  <- llrem:::get_base_riskset(NULL, seq_nodes)

      sm_llm <- tryCatch(
        compute_seq_stat_means(sim$sequence, rem_formula,
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

# Domain-level means (average stat means across networks within each domain)
domain_means <- aggregate(mean_val ~ domain + source + provider + term + valence + visibility,
                           data = stat_df, FUN = mean, na.rm = TRUE)

# --- bias_df: signed deviation from empirical, at domain level ---
emp_domain <- domain_means[domain_means$source == "Empirical",
                            c("domain", "term", "mean_val")]
names(emp_domain)[names(emp_domain) == "mean_val"] <- "emp_val"

llm_domain <- domain_means[domain_means$source != "Empirical", ]

bias_df <- merge(llm_domain, emp_domain, by = c("domain", "term"))
bias_df$bias <- bias_df$mean_val - bias_df$emp_val

# --- rmse_df: per network and domain-level average ---
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
      sim     <- results[[run_key]]
      if (is.null(sim)) next

      seq_nodes   <- sort(unique(c(sim$sequence$sender, sim$sequence$receiver)))
      seq_riskset <- llrem:::get_base_riskset(NULL, seq_nodes)

      sm_llm <- tryCatch(
        compute_seq_stat_means(sim$sequence, rem_formula,
                                seq_riskset)[stat_terms],
        error = function(e) NULL
      )
      if (is.null(sm_llm)) next

      rmse <- sqrt(mean((sm_llm - sm_emp)^2, na.rm = TRUE))
      rmse_rows[[length(rmse_rows) + 1L]] <- data.frame(
        domain     = domain_id,
        net_key    = net_key,
        net_idx    = net_idx,
        provider   = provider_id,
        source     = providers[[provider_id]]$name,
        RMSE       = round(rmse, 3),
        stringsAsFactors = FALSE
      )
    }
  }
}

rmse_df <- do.call(rbind, rmse_rows)
rmse_df <- rmse_df[order(rmse_df$domain, rmse_df$RMSE), ]

# Domain-level average RMSE across networks
rmse_domain <- aggregate(RMSE ~ domain + provider + source,
                          data = rmse_df, FUN = mean, na.rm = TRUE)
rmse_domain$RMSE <- round(rmse_domain$RMSE, 3)

cat("RMSE by domain and provider (averaged across networks):\n")
print(rmse_domain[order(rmse_domain$domain, rmse_domain$RMSE), ], row.names = FALSE)

# =============================================================================
# 9. PLOTS
# =============================================================================

provider_palette <- c(
  "Empirical"                  = "#2166AC",
  "Bedrock/Claude (Haiku 4.5)" = "#1B7837",
  "Bedrock/Llama (3.1 70B)"    = "#762A83"
)

# --- Plot 1: Domain × stat dot plot (9 facets, one per term) ----------------
# Uses domain-level means. Diamond = empirical; circle/triangle = LLM.

emp_dm  <- domain_means[domain_means$source == "Empirical", ]
llm_dm  <- domain_means[domain_means$source != "Empirical", ]

shape_vals <- c(
  "Empirical"                  = 18L,
  setNames(c(16L, 17L), vapply(providers, `[[`, character(1L), "name"))
)

p_suffstat <- ggplot(mapping = aes(x = mean_val, y = domain)) +
  geom_segment(data = llm_dm,
               aes(x = -Inf, xend = mean_val, y = domain, yend = domain,
                   group = interaction(domain, source)),
               colour = "grey72", linewidth = 0.3) +
  geom_segment(data = emp_dm,
               aes(x = -Inf, xend = mean_val, y = domain, yend = domain),
               colour = "#2166AC", linewidth = 0.3, alpha = 0.45) +
  geom_point(data = llm_dm,
             aes(colour = source, shape = source),
             size = 3, alpha = 0.7) +
  geom_point(data = emp_dm,
             aes(colour = source, shape = source),
             size = 3.5, alpha = 0.9) +
  scale_colour_manual(values = provider_palette) +
  scale_shape_manual(values = shape_vals) +
  facet_wrap(~ term, scales = "free_x", nrow = 3L) +
  labs(
    x       = "Mean covariate value across realized events (domain average)",
    y       = NULL,
    colour  = NULL,
    shape   = NULL,
    title   = "Sufficient statistic means by domain — multi-agent simulation",
    caption = "Diamond = empirical baseline. Points are averages across networks within each domain."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position   = "bottom",
    strip.text        = element_text(size = 9, face = "bold"),
    strip.background  = element_rect(fill = "grey92", colour = "grey60", linewidth = 0.5),
    panel.border      = element_rect(colour = "grey60", fill = NA, linewidth = 0.5),
    panel.grid.major.y = element_blank(),
    panel.grid.minor  = element_blank(),
    panel.spacing     = unit(1, "lines")
  )

print(p_suffstat)
ggsave(file.path(STUDY_DIR, "multiagent_multidomain_suffstat.pdf"),
       p_suffstat, width = 14, height = 9)

# --- Plot 2: Signed-bias heatmap (rows = terms, cols = domains) -------------
# The primary plot for detecting systematic LLM biases.

bias_df$term   <- factor(bias_df$term,   levels = rev(stat_terms))
bias_df$domain <- factor(bias_df$domain, levels = ACTIVE_DOMAINS)

bias_limit <- max(abs(bias_df$bias), na.rm = TRUE)

p_bias <- ggplot(bias_df, aes(x = domain, y = term, fill = bias)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%+.2f", bias)),
            size = 2.6, colour = "grey20") +
  scale_fill_gradient2(
    low      = "#D6604D",
    mid      = "white",
    high     = "#4393C3",
    midpoint = 0,
    limits   = c(-bias_limit, bias_limit),
    name     = "LLM − empirical"
  ) +
  facet_wrap(~ source, ncol = 1L) +
  labs(
    x       = NULL,
    y       = NULL,
    title   = "LLM bias by domain and sufficient statistic term",
    subtitle = "Fill = mean(LLM) − mean(empirical); red = under-expression, blue = over-expression",
    caption = "Values are domain averages across networks."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x     = element_text(angle = 35, hjust = 1),
    strip.text      = element_text(size = 9, face = "bold"),
    strip.background = element_rect(fill = "grey92", colour = "grey60", linewidth = 0.5),
    panel.border    = element_rect(colour = "grey60", fill = NA, linewidth = 0.5),
    panel.grid      = element_blank(),
    legend.position = "right"
  )

print(p_bias)
ggsave(file.path(STUDY_DIR, "multiagent_multidomain_bias.pdf"),
       p_bias, width = 10, height = 7)

# --- Plot 3: RMSE bar chart by domain × provider ----------------------------

rmse_domain$domain   <- factor(rmse_domain$domain,   levels = ACTIVE_DOMAINS)
rmse_domain$provider <- factor(rmse_domain$provider, levels = ACTIVE_PROVIDERS)

p_rmse <- ggplot(rmse_domain, aes(x = domain, y = RMSE, fill = source)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, alpha = 0.85) +
  scale_fill_manual(values = c(
    "Bedrock/Claude (Haiku 4.5)" = "#1B7837",
    "Bedrock/Llama (3.1 70B)"   = "#762A83"
  )) +
  labs(
    x       = NULL,
    y       = "RMSE vs empirical sufficient statistic means",
    fill    = NULL,
    title   = "RMSE by domain — multi-agent simulation",
    caption = "RMSE is averaged across networks within each domain."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x     = element_text(angle = 35, hjust = 1),
    legend.position = "bottom",
    panel.grid.major.x = element_blank()
  )

print(p_rmse)
ggsave(file.path(STUDY_DIR, "multiagent_multidomain_rmse.pdf"),
       p_rmse, width = 8, height = 5)

# =============================================================================
# 10. SAVE
# =============================================================================

save(results, emp_rems, emp_sms, stat_df, domain_means,
     bias_df, rmse_df, rmse_domain,
     file = file.path(STUDY_DIR, "multiagent_multidomain_results.RData"))

cat(sprintf("\nResults saved to %s\n",
            file.path(STUDY_DIR, "multiagent_multidomain_results.RData")))
cat(sprintf("Plots saved to %s/multiagent_multidomain_*.pdf\n", STUDY_DIR))
