# global.R — runs once at app startup (shared across all sessions)

suppressPackageStartupMessages({
  library(survival)
  library(ggplot2)
  library(shiny)
  library(promises)
  library(future)
})

devtools::load_all("../llrem", quiet = TRUE)

# Background execution for ExtendedTask
future::plan(future::multisession, workers = 2)

# ------------------------------------------------------------------
# AWS credentials: read from environment / .Renviron only
# ------------------------------------------------------------------
if (!nzchar(Sys.getenv("AWS_ACCESS_KEY_ID"))) {
  warning("AWS_ACCESS_KEY_ID not set — Bedrock provider will fail at runtime.")
}

# ------------------------------------------------------------------
# Empirical data + REM (fitted once)
# ------------------------------------------------------------------
data("edgelist", package = "llrem")
EDGELIST     <- prepare_edgelist(edgelist)
ALL_NODES    <- sort(unique(c(EDGELIST$sender, EDGELIST$receiver)))
BASE_RISKSET <- llrem:::get_base_riskset(NULL, ALL_NODES)

REM_FORMULA <- survival::Surv(time = time1, event = occurred) ~
  Imitation + Repetition + RRecInitiation + DyadRecencyLog + IOviolation + NTriadic +
  strata(time2)

message("[wargame-ui] Fitting empirical REM...")
REM_EMPIRICAL <- fit_rem(EDGELIST, REM_FORMULA,
                         n            = length(ALL_NODES),
                         n_nonevents  = 200L,
                         base_riskset = BASE_RISKSET)
message("[wargame-ui] Empirical REM ready.")

EMPIRICAL_COEF <- coef(REM_EMPIRICAL)
STAT_TERMS     <- c("Imitation", "Repetition", "RRecInitiation",
                    "DyadRecencyLog", "IOviolation", "NTriadic")

CONFLICT_CONTEXT <- paste(
  "You are participating in a simulation of an international militarized",
  "conflict network. Actors are sovereign countries. Each event represents",
  "one country using military force against another."
)

CHECKPOINT_DIR <- file.path("~", "Projects", "LLMREM", "checkpoints_wargame")
dir.create(CHECKPOINT_DIR, showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------------
# Helpers shared by server and background worker
# ------------------------------------------------------------------

# Parse a comma-separated string of integers into a sorted integer vector.
parse_int_list <- function(s) {
  vals <- suppressWarnings(as.integer(trimws(strsplit(s, "[,\\s]+")[[1L]])))
  sort(unique(vals[!is.na(vals)]))
}

# Read the latest tick written to a checkpoint file (without loading everything).
checkpoint_tick <- function(path) {
  if (!file.exists(path)) return(0L)
  tryCatch({
    env <- new.env(parent = emptyenv())
    load(path, envir = env)
    if (exists("next_tick", envir = env)) as.integer(env$next_tick) - 1L else 0L
  }, error = function(e) 0L)
}

# Load a completed or in-progress checkpoint; return NULL on failure.
load_checkpoint <- function(path) {
  if (!file.exists(path)) return(NULL)
  tryCatch({
    env <- new.env(parent = emptyenv())
    load(path, envir = env)
    as.list(env)
  }, error = function(e) NULL)
}
