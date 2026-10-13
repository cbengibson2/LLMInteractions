# =============================================================================
# run_reasoning_probe_multidomain.R
#
# Cross-domain reasoning probe. For each interaction ecology, we replay
# 10 high-endogeneity events from the Claude sender_obs sequence and ask:
#   1. Free-text: explain your reasoning before choosing
#   2. Integer-only: confirm the choice
#
# The dissociation of interest: does the model narrate strategic/contextual
# reasoning while producing a statistically endogenous behavioural choice?
#
# The endogenous keyword list is universal. The strategic keyword list is
# domain-specific — different ecologies prime different deliberative frames.
# =============================================================================

devtools::load_all("llrem", quiet = TRUE)
library(survival)

# =============================================================================
# 1. CREDENTIALS
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
# 2. DOMAIN CONFIGURATIONS
# =============================================================================

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root
DATA_DIR  <- file.path(STUDY_DIR, "DataP")

DOMAIN_CONFIGS <- list(

  War = list(
    data_path   = file.path(DATA_DIR, "War/conflict.Rdata"),
    n_events    = 1000L,
    context     = paste(
      "You are participating in a simulation of an international militarized",
      "conflict network. Actors are sovereign countries. Each event represents",
      "one country using military force against another."
    ),
    action_verb = "attacked",
    actor_noun  = "Country",
    strategic_terms = c("power", "strateg", "weak", "strong", "alliance",
                        "interest", "opportun", "threat", "capabilit",
                        "econom", "geopolit", "leverage", "vulnerab",
                        "border", "militar", "regime", "sanction")
  ),

  Classroom = list(
    data_path   = file.path(DATA_DIR, "Classroom/McFarland.Rdata"),
    n_events    = NULL,
    context     = paste(
      "You are participating in a simulation of spoken interaction in a",
      "high-school classroom. Actors are students and a teacher. Each event",
      "represents one participant directing a spoken utterance to another.",
      "Node 21 represents the class as a whole (group address)."
    ),
    action_verb = "spoke to",
    actor_noun  = "Participant",
    strategic_terms = c("popular", "teacher", "influenc", "status", "authorit",
                        "dominan", "attent", "respect", "knowledge", "engag",
                        "impress", "smart", "social", "friend", "hierarch")
  ),

  Gangs = list(
    data_path   = file.path(DATA_DIR, "Gangs/gangnet1.Rdata"),
    n_events    = NULL,
    context     = paste(
      "You are participating in a simulation of gang violence in an urban",
      "neighbourhood. Actors are street gangs. Each event represents one gang",
      "carrying out a violent act against another gang."
    ),
    action_verb = "attacked",
    actor_noun  = "Gang",
    strategic_terms = c("territor", "dominan", "rival", "turf", "crew",
                        "strong", "weak", "threat", "power", "control",
                        "oppos", "target", "vulnerab", "opportun", "resource")
  ),

  Email = list(
    data_path   = file.path(DATA_DIR, "Email/eu1.txt1.Rdata"),
    n_events    = 1000L,
    context     = paste(
      "You are participating in a simulation of email communication within a",
      "research organisation. Actors are employees. Each event represents one",
      "employee sending an email to another employee."
    ),
    action_verb = "emailed",
    actor_noun  = "Employee",
    strategic_terms = c("hierarch", "senior", "manag", "authorit", "influenc",
                        "collaborat", "expert", "project", "deadline", "report",
                        "promot", "visibl", "network", "opportun", "import")
  ),

  Twitter = list(
    data_path   = file.path(DATA_DIR, "Twitter/twitter_boston_relevent.rdata"),
    n_events    = NULL,
    context     = paste(
      "You are participating in a simulation of a Twitter conversation.",
      "Actors are users. Each event represents one user sending a public",
      "reply or mention to another user."
    ),
    action_verb = "replied to",
    actor_noun  = "User",
    strategic_terms = c("popular", "influenc", "follower", "engag", "viral",
                        "reach", "audienc", "trend", "retweet", "amplif",
                        "visib", "platform", "promot", "opportun", "credib")
  ),

  Phone = list(
    data_path   = file.path(DATA_DIR, "Phone/Phone.Rdata"),
    n_events    = NULL,
    context     = paste(
      "You are participating in a simulation of phone calls during a military",
      "command-and-control exercise. Actors are organisational roles. Each",
      "event represents one role placing a call to another role."
    ),
    action_verb = "called",
    actor_noun  = "Role",
    strategic_terms = c("authorit", "command", "rank", "superior", "report",
                        "mission", "chain", "hierarch", "coordinat", "briefing",
                        "priority", "urgent", "decision", "respons", "protocol")
  )

)

ACTIVE_DOMAINS <- c("War", "Classroom", "Gangs", "Email", "Twitter", "Phone")

# =============================================================================
# 3. SHARED UTILITIES
# =============================================================================

rem_formula <- Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

# Endogenous terms are domain-invariant
ENDOGENOUS_TERMS <- c("history", "previous", "before", "past", "repeated",
                      "again", "already", "familiar", "prior", "last time",
                      "pattern", "recent", "retaliat", "reciproc", "habit",
                      "usual", "typical", "same", "again", "established")

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

classify_reasoning <- function(text, strategic_terms) {
  tl   <- tolower(text)
  n_e  <- sum(vapply(ENDOGENOUS_TERMS, grepl, logical(1L), x = tl))
  n_s  <- sum(vapply(strategic_terms,  grepl, logical(1L), x = tl))
  if (n_e > n_s)      "endogenous"
  else if (n_s > n_e) "strategic"
  else                "mixed"
}

probe_event <- function(t, cfg, seq_df, all_nodes, pe) {
  past      <- seq_df[seq_len(t), ]
  evt       <- seq_df[t + 1L, ]
  sender_id <- evt$sender
  valid_targets <- setdiff(all_nodes, sender_id)

  history_str  <- llrem:::format_history(utils::tail(past, cfg$history_window), cfg)
  sender_label <- llrem:::node_label(sender_id, cfg)
  target_str   <- paste(
    vapply(valid_targets,
           function(v) sprintf("%s (ID: %d)", llrem:::node_label(v, cfg), v),
           character(1L)),
    collapse = ", "
  )

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

  raw_reason <- llm_call(cfg$provider, msgs_reason, max_tokens = 200L,
                          temperature = 0.7)
  raw_choice <- llm_call(cfg$provider, msgs_choice, max_tokens = 15L,
                          temperature = 0.7)

  chosen     <- llrem:::first_valid_int(raw_choice, valid_targets)
  reasoned_id <- llrem:::first_valid_int(
    utils::tail(strsplit(raw_reason, "\n")[[1L]], 1L),
    valid_targets
  )

  pe_row <- pe[pe$t == t, ]
  list(
    t               = t,
    sender          = sender_id,
    sender_label    = sender_label,
    actual_receiver = evt$receiver,
    Repetition      = if (nrow(pe_row) > 0L) pe_row$Repetition[1L]    else NA,
    RRecInitiation   = if (nrow(pe_row) > 0L) pe_row$RRecInitiation[1L] else NA,
    DyadRecencyLog      = if (nrow(pe_row) > 0L) pe_row$DyadRecencyLog[1L]    else NA,
    endogeneity_score = if (nrow(pe_row) > 0L) pe_row$endogeneity_score[1L] else NA,
    reasoning       = raw_reason,
    reasoned_id     = reasoned_id,
    choice_id       = chosen
  )
}

# =============================================================================
# 4. MAIN LOOP
# =============================================================================

all_probe_results <- list()

for (domain_id in ACTIVE_DOMAINS) {
  cat(sprintf("\n%s\n%s\n", strrep("=", 70), domain_id))

  dcfg <- DOMAIN_CONFIGS[[domain_id]]

  # --- Load data and reconstruct sender_obs sequence from checkpoint ---
  dat      <- load_datap(dcfg$data_path)
  edgelist <- dat$edgelist
  n_ev     <- if (is.null(dcfg$n_events)) nrow(edgelist) else
                min(dcfg$n_events, nrow(edgelist))
  all_nodes    <- sort(unique(c(edgelist$sender, edgelist$receiver)))
  base_riskset <- llrem:::get_base_riskset(NULL, all_nodes)

  ckpt_path <- file.path(STUDY_DIR,
    sprintf("checkpoints_multidomain/bedrock_claude_%s_sender_obs.RData", domain_id))

  if (!file.exists(ckpt_path)) {
    message("  [SKIP] No checkpoint found: ", ckpt_path)
    next
  }

  ckpt_env <- new.env(parent = emptyenv())
  load(ckpt_path, envir = ckpt_env)
  seq_sender <- rbind(
    edgelist[1L, c("time", "sender", "receiver", "eveid")],
    do.call(rbind, ckpt_env$llm_rows)
  )
  cat(sprintf("  Sequence: %d events  |  Actors: %d\n",
              nrow(seq_sender), length(all_nodes)))

  # --- Per-event sufficient statistics ---
  pe <- tryCatch(
    llrem:::remstats_generate(seq_sender, n = length(all_nodes),
                               n_nonevents  = 0L,
                               base_riskset = base_riskset),
    error = function(e) { message("  [WARN] remstats failed: ", e$message); NULL }
  )
  if (is.null(pe)) next
  pe <- llrem:::add_survival_cols(pe)
  pe$t <- seq_len(nrow(pe))

  pe$endogeneity_score <- pe$Repetition +
    pe$RRecInitiation +
    abs(pe$DyadRecencyLog) / max(abs(pe$DyadRecencyLog), na.rm = TRUE)

  # --- Sample events across endogeneity quartiles ---
  set.seed(42L)
  q75 <- quantile(pe$endogeneity_score, 0.75, na.rm = TRUE)
  q50 <- quantile(pe$endogeneity_score, 0.50, na.rm = TRUE)
  q25 <- quantile(pe$endogeneity_score, 0.25, na.rm = TRUE)

  top <- pe[!is.na(pe$endogeneity_score) & pe$endogeneity_score >= q75, ]
  mid <- pe[!is.na(pe$endogeneity_score) & pe$endogeneity_score >= q50 &
              pe$endogeneity_score < q75, ]
  low <- pe[!is.na(pe$endogeneity_score) & pe$endogeneity_score >= q25 &
              pe$endogeneity_score < q50, ]

  # Fewer events in smaller sequences — scale sample sizes accordingly
  n_top <- min(5L, nrow(top))
  n_mid <- min(3L, nrow(mid))
  n_low <- min(2L, nrow(low))

  sampled <- rbind(
    if (n_top > 0L) top[sample(nrow(top), n_top), ] else top[0L, ],
    if (n_mid > 0L) mid[sample(nrow(mid), n_mid), ] else mid[0L, ],
    if (n_low > 0L) low[sample(nrow(low), n_low), ] else low[0L, ]
  )
  sampled <- sampled[order(sampled$t), ]
  # Drop any t where seq_sender[t+1] doesn't exist
  sampled <- sampled[sampled$t < nrow(seq_sender), ]

  cat(sprintf("  Probing %d events (score range: %.2f – %.2f)\n",
              nrow(sampled),
              min(sampled$endogeneity_score, na.rm = TRUE),
              max(sampled$endogeneity_score, na.rm = TRUE)))

  # --- Build cfg ---
  cfg <- rem_cfg(
    provider       = probe_provider,
    context        = dcfg$context,
    action_verb    = dcfg$action_verb,
    actor_noun     = dcfg$actor_noun,
    node_labels    = NULL,
    history_window = 30L
  )

  # --- Run probes ---
  domain_results <- vector("list", nrow(sampled))

  for (i in seq_len(nrow(sampled))) {
    t <- sampled$t[i]
    cat(sprintf("  [%d/%d] t=%d  Repetition=%.2f  RecInit=%.2f  DyadRecencyLog=%.2f\n",
                i, nrow(sampled), t,
                sampled$Repetition[i],
                sampled$RRecInitiation[i],
                sampled$DyadRecencyLog[i]))

    domain_results[[i]] <- tryCatch(
      probe_event(t, cfg, seq_sender, all_nodes, pe),
      error = function(e) {
        message("    Error: ", e$message); NULL
      }
    )

    if (!is.null(domain_results[[i]])) {
      r <- domain_results[[i]]
      cls <- classify_reasoning(r$reasoning, dcfg$strategic_terms)
      diss <- !is.na(r$reasoned_id) && !is.na(r$choice_id) &&
                r$reasoned_id != r$choice_id
      cat(sprintf("    Sender: %s  actual→%d  re-probed→%d  reasoned→%d  [%s]%s\n",
                  r$sender_label, r$actual_receiver,
                  ifelse(is.na(r$choice_id), -1L, r$choice_id),
                  ifelse(is.na(r$reasoned_id), -1L, r$reasoned_id),
                  cls,
                  if (diss) "  *** DISSOCIATION ***" else ""))
      cat(sprintf("    Reasoning: %s\n",
                  substr(gsub("\n", " ", r$reasoning), 1L, 120L)))
    }
    Sys.sleep(1)
  }

  domain_results <- Filter(Negate(is.null), domain_results)

  # Classify and summarise
  classes <- vapply(domain_results,
                    function(r) classify_reasoning(r$reasoning, dcfg$strategic_terms),
                    character(1L))
  diss_flags <- vapply(domain_results, function(r)
    !is.na(r$reasoned_id) && !is.na(r$choice_id) && r$reasoned_id != r$choice_id,
    logical(1L))

  cat(sprintf("\n  === %s summary ===\n", domain_id))
  print(table(classes))
  cat(sprintf("  Dissociations (reasoned ≠ chosen): %d / %d\n",
              sum(diss_flags), length(domain_results)))

  all_probe_results[[domain_id]] <- list(
    results    = domain_results,
    classes    = classes,
    diss_flags = diss_flags
  )
}

# =============================================================================
# 5. CROSS-DOMAIN SUMMARY
# =============================================================================

cat("\n\n=== CROSS-DOMAIN REASONING PROBE SUMMARY ===\n\n")

summary_rows <- do.call(rbind, lapply(names(all_probe_results), function(dom) {
  d <- all_probe_results[[dom]]
  data.frame(
    Domain         = dom,
    N              = length(d$results),
    Endogenous     = sum(d$classes == "endogenous"),
    Strategic      = sum(d$classes == "strategic"),
    Mixed          = sum(d$classes == "mixed"),
    Dissociations  = sum(d$diss_flags),
    Pct_endo       = round(100 * mean(d$classes == "endogenous")),
    Pct_diss       = round(100 * mean(d$diss_flags)),
    stringsAsFactors = FALSE
  )
}))
print(summary_rows, row.names = FALSE)

# =============================================================================
# 6. SAVE
# =============================================================================

save(all_probe_results, summary_rows,
     file = file.path(STUDY_DIR, "probe_results_multidomain.RData"))
cat(sprintf("\nSaved to %s\n",
            file.path(STUDY_DIR, "probe_results_multidomain.RData")))
