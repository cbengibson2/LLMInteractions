# app.R — Wargame Scenario Explorer
#
# Layout:
#   Sidebar  — scenario settings, intervention builder, run controls
#   Main     — four tabs:
#               1. Live feed   (event stream + progress bar)
#               2. Activation  (sender p_i(t) heatmap, live)
#               3. Results     (coefficient forest plot + RMSE, post-run)
#               4. Intervention log (table of fired interventions)

source("global.R")

# ============================================================
# UI
# ============================================================

ui <- fluidPage(
  title = "Wargame Scenario Explorer",
  theme = bslib::bs_theme(bootswatch = "flatly", base_font_size = "14px"),

  tags$head(tags$style(HTML("
    .sidebar-panel { padding-top: 12px; }
    .iv-card { border: 1px solid #dee2e6; border-radius: 6px;
               padding: 8px 10px; margin-bottom: 6px; background: #f8f9fa; }
    .status-pill { font-size: 0.8em; padding: 2px 8px; border-radius: 10px; }
    .status-idle    { background:#e9ecef; color:#495057; }
    .status-running { background:#cfe2ff; color:#084298; }
    .status-done    { background:#d1e7dd; color:#0f5132; }
    .status-error   { background:#f8d7da; color:#842029; }
    pre.event-feed  { max-height: 380px; overflow-y: auto;
                      font-size: 0.82em; background: #212529; color: #adb5bd;
                      padding: 10px; border-radius: 4px; }
  "))),

  titlePanel(
    fluidRow(
      column(9, "Wargame Scenario Explorer"),
      column(3, align = "right",
        div(style = "margin-top: 8px;",
          uiOutput("status_pill"))
      )
    )
  ),

  sidebarLayout(
    # ----------------------------------------------------------
    # SIDEBAR
    # ----------------------------------------------------------
    sidebarPanel(
      width = 3,
      class = "sidebar-panel",

      # --- LLM Provider ---
      h5("LLM Provider"),
      selectInput("model", "Bedrock model",
        choices = c(
          "Claude Haiku 4.5"  = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
          "Claude Sonnet 4.6" = "us.anthropic.claude-sonnet-4-6-20250514-v1:0"
        )
      ),

      hr(),

      # --- Simulation settings ---
      h5("Simulation"),
      sliderInput("n_ticks", "Ticks to simulate", 50L, 1000L, 200L, step = 50L),
      numericInput("seed", "RNG seed", 42L, min = 1L),
      checkboxInput("sequential", "Sequential moves within tick", TRUE),
      selectInput("behavioral_frame", "Behavioural frame",
        c("Default" = "default",
          "Endogenous (retaliation)" = "endogenous",
          "Deliberative (strategic)" = "deliberative")
      ),

      hr(),

      # --- Interventions ---
      h5("Add intervention"),

      selectInput("iv_type", "Type",
        c("Military strike"   = "military",
          "Ceasefire"         = "ceasefire",
          "Third-party entry" = "entry")
      ),

      conditionalPanel("input.iv_type === 'military'",
        numericInput("iv_mil_tick",     "At tick",    50L, 1L, 1000L),
        numericInput("iv_mil_sender",   "Sender ID",   3L, 1L,  200L),
        numericInput("iv_mil_receiver", "Receiver ID", 7L, 1L,  200L)
      ),

      conditionalPanel("input.iv_type === 'ceasefire'",
        numericInput("iv_cf_start",   "From tick",  50L, 1L, 1000L),
        numericInput("iv_cf_end",     "To tick",   100L, 1L, 1000L),
        textInput("iv_cf_dyads",
          "Dyads (sender,receiver; one per line or comma-separated)",
          "3,7\n7,3"),
        helpText("Each line is one sender→receiver pair to suppress.")
      ),

      conditionalPanel("input.iv_type === 'entry'",
        numericInput("iv_en_tick",     "At tick",        55L, 1L, 1000L),
        numericInput("iv_en_actor",    "New actor ID",   96L, 1L,  999L),
        textInput(   "iv_en_label",    "Actor label",    "Mediator"),
        textInput(   "iv_en_targets",  "Initial targets (comma-sep IDs)", "3,7")
      ),

      actionButton("add_iv", "Add intervention", class = "btn-sm btn-secondary",
                   width = "100%"),

      tags$div(id = "iv_list_container",
        uiOutput("iv_list")
      ),

      hr(),

      # --- Run controls ---
      actionButton("run_btn", "Run scenario",
                   class = "btn-primary", width = "100%"),
      br(), br(),
      actionButton("clear_btn", "Clear & reset",
                   class = "btn-sm btn-outline-danger", width = "100%")
    ),

    # ----------------------------------------------------------
    # MAIN PANEL
    # ----------------------------------------------------------
    mainPanel(
      width = 9,

      tabsetPanel(
        id = "main_tabs",

        # Tab 1: Live feed
        tabPanel("Live feed",
          br(),
          fluidRow(
            column(4, uiOutput("box_ticks")),
            column(4, uiOutput("box_events")),
            column(4, uiOutput("box_ivs"))
          ),
          br(),
          div(style = "margin-bottom: 6px;",
            strong("Progress"), br(),
            shinyWidgets::progressBar(
              id       = "progress_bar",
              value    = 0,
              total    = 100,
              display_pct = TRUE,
              status   = "info"
            )
          ),
          br(),
          strong("Event stream (most recent first)"),
          verbatimTextOutput("event_feed") |>
            tagAppendAttributes(class = "event-feed")
        ),

        # Tab 2: Sender activation heatmap
        tabPanel("Activation",
          br(),
          helpText("Sender activation probability p_i(t) for the 15 most-active ",
                   "actors. Updated from checkpoint every 5 s while running."),
          plotOutput("activation_plot", height = "480px")
        ),

        # Tab 3: Results (post-run)
        tabPanel("Results",
          br(),
          helpText("Coefficient forest plot and RMSE available after the simulation completes."),
          plotOutput("coef_plot", height = "400px"),
          br(),
          tableOutput("rmse_table")
        ),

        # Tab 4: Intervention log
        tabPanel("Intervention log",
          br(),
          tableOutput("iv_log_table")
        )
      )
    )
  )
)

# ============================================================
# SERVER
# ============================================================

server <- function(input, output, session) {

  # ---- Reactive state -----------------------------------------------
  interventions_rv <- reactiveVal(list())    # list of intervention objects
  sim_result_rv    <- reactiveVal(NULL)      # final result (from ExtendedTask)
  run_status_rv    <- reactiveVal("idle")    # "idle" | "running" | "done" | "error"
  checkpoint_path  <- reactiveVal(NULL)      # path to active checkpoint file
  poll_data_rv     <- reactiveVal(NULL)      # partial data from checkpoint polling

  # ---- Scenario ID (used for checkpoint filename) -------------------
  scenario_id <- reactive({
    paste0("session_", substr(session$token, 1L, 8L))
  })

  # ---- Add intervention ---------------------------------------------
  observeEvent(input$add_iv, {
    iv <- switch(input$iv_type,
      military = intervention_military(
        tick     = as.integer(input$iv_mil_tick),
        sender   = as.integer(input$iv_mil_sender),
        receiver = as.integer(input$iv_mil_receiver)
      ),
      ceasefire = {
        raw_lines <- strsplit(input$iv_cf_dyads, "[\\n,]+")[[1L]]
        raw_lines <- trimws(raw_lines[nzchar(trimws(raw_lines))])
        pairs <- lapply(raw_lines, function(ln) {
          v <- suppressWarnings(as.integer(trimws(strsplit(ln, "[,\\s]+")[[1L]])))
          v[!is.na(v)][1:2]
        })
        pairs <- Filter(function(p) length(p) == 2L && !any(is.na(p)), pairs)
        if (length(pairs) == 0L) {
          showNotification("No valid dyads found.", type = "warning")
          return()
        }
        intervention_ceasefire(
          tick_start = as.integer(input$iv_cf_start),
          tick_end   = as.integer(input$iv_cf_end),
          dyads      = pairs
        )
      },
      entry = {
        targets <- parse_int_list(input$iv_en_targets)
        intervention_entry(
          tick            = as.integer(input$iv_en_tick),
          actor_id        = as.integer(input$iv_en_actor),
          actor_label     = trimws(input$iv_en_label),
          initial_targets = if (length(targets) > 0L) targets else NULL
        )
      }
    )
    interventions_rv(c(interventions_rv(), list(iv)))
    showNotification(paste("Added:", iv$label), type = "message", duration = 2)
  })

  # Remove an intervention by index
  observeEvent(input$remove_iv, {
    idx <- as.integer(input$remove_iv)
    ivs <- interventions_rv()
    if (idx >= 1L && idx <= length(ivs))
      interventions_rv(ivs[-idx])
  })

  # ---- Render intervention list -------------------------------------
  output$iv_list <- renderUI({
    ivs <- interventions_rv()
    if (length(ivs) == 0L)
      return(helpText("No interventions added yet."))
    tags$div(
      lapply(seq_along(ivs), function(i) {
        iv <- ivs[[i]]
        badge_col <- switch(iv$type,
          military  = "#f8d7da",
          ceasefire = "#fff3cd",
          entry     = "#d1e7dd",
          "#e9ecef"
        )
        div(class = "iv-card",
          style = paste0("border-left: 4px solid ", badge_col, ";"),
          fluidRow(
            column(9,
              tags$small(tags$b(toupper(iv$type)), " — ", iv$label)
            ),
            column(3, align = "right",
              actionButton(
                inputId = "remove_iv",
                label   = "×",
                class   = "btn-sm btn-link text-danger p-0",
                onclick = paste0("Shiny.setInputValue('remove_iv', ", i,
                                 ", {priority: 'event'})")
              )
            )
          )
        )
      })
    )
  })

  # ---- ExtendedTask: run simulation in background ------------------
  sim_task <- ExtendedTask$new(function(edgelist, rem_fit, cfg, wgcfg,
                                        interventions, n_ticks, seed,
                                        ckpt_path) {
    future::future({
      run_wargame(
        edgelist        = edgelist,
        rem_fit         = rem_fit,
        cfg             = cfg,
        n_ticks         = n_ticks,
        interventions   = interventions,
        wgcfg           = wgcfg,
        lambda          = NULL,
        seed            = seed,
        checkpoint_path = ckpt_path
      )
    }, seed = TRUE)
  })

  # ---- Run button ---------------------------------------------------
  observeEvent(input$run_btn, {
    req(run_status_rv() %in% c("idle", "done", "error"))

    # Build provider & configs
    provider <- llm_provider_bedrock(model = input$model)
    cfg      <- rem_cfg(
      provider       = provider,
      context        = CONFLICT_CONTEXT,
      action_verb    = "attacked",
      actor_noun     = "Country",
      history_window = 30L
    )
    wgcfg <- wargame_cfg(
      sequential       = input$sequential,
      behavioral_frame = input$behavioral_frame
    )

    ckpt <- file.path(CHECKPOINT_DIR,
                      paste0(scenario_id(), "_", format(Sys.time(), "%H%M%S"),
                             ".RData"))
    checkpoint_path(ckpt)
    sim_result_rv(NULL)
    poll_data_rv(NULL)
    run_status_rv("running")

    shinyWidgets::updateProgressBar(session, "progress_bar",
                                    value = 0, total = input$n_ticks)

    sim_task$invoke(EDGELIST, REM_EMPIRICAL, cfg, wgcfg,
                    interventions_rv(), as.integer(input$n_ticks),
                    as.integer(input$seed), ckpt)
  })

  # Collect task result when done
  observe({
    req(run_status_rv() == "running")
    status <- sim_task$status()
    if (status == "success") {
      result <- sim_task$result()
      sim_result_rv(result)
      run_status_rv("done")
      shinyWidgets::updateProgressBar(session, "progress_bar",
                                      value = input$n_ticks,
                                      total = input$n_ticks)
    } else if (status == "error") {
      run_status_rv("error")
      showNotification(
        paste("Simulation error:", conditionMessage(sim_task$result())),
        type = "error", duration = 10
      )
    }
  })

  # ---- Clear / reset ------------------------------------------------
  observeEvent(input$clear_btn, {
    req(run_status_rv() != "running")
    interventions_rv(list())
    sim_result_rv(NULL)
    poll_data_rv(NULL)
    run_status_rv("idle")
    checkpoint_path(NULL)
    shinyWidgets::updateProgressBar(session, "progress_bar",
                                    value = 0, total = 100)
  })

  # ---- Checkpoint polling (every 5 s while running) -----------------
  checkpoint_timer <- reactiveTimer(5000)

  observe({
    checkpoint_timer()
    req(run_status_rv() == "running")
    ckpt <- checkpoint_path()
    req(!is.null(ckpt))

    data <- load_checkpoint(ckpt)
    if (!is.null(data)) {
      poll_data_rv(data)
      tick_done <- checkpoint_tick(ckpt)
      shinyWidgets::updateProgressBar(session, "progress_bar",
                                      value    = tick_done,
                                      total    = input$n_ticks)
    }
  })

  # ---- Status pill --------------------------------------------------
  output$status_pill <- renderUI({
    s <- run_status_rv()
    cls <- paste("status-pill", paste0("status-", s))
    label <- switch(s,
      idle    = "Idle",
      running = "Running...",
      done    = "Complete",
      error   = "Error"
    )
    span(class = cls, label)
  })

  # ---- Helper: latest data (final > polling) ------------------------
  current_data <- reactive({
    if (!is.null(sim_result_rv())) return(sim_result_rv())
    poll_data_rv()
  })

  # ---- Stat cards (plain bslib cards, no shinydashboard needed) ----
  stat_card <- function(value, label, bg = "#e9f0fb") {
    div(style = paste0(
          "background:", bg, "; border-radius:6px; padding:12px 16px; ",
          "text-align:center; margin-bottom:10px;"),
      div(style = "font-size:2em; font-weight:700; line-height:1.1;", value),
      div(style = "font-size:0.85em; color:#555; margin-top:2px;",    label)
    )
  }

  output$box_ticks <- renderUI({
    d <- current_data()
    n <- if (!is.null(d)) {
      if (!is.null(d$next_tick)) d$next_tick - 1L
      else if (!is.null(d$n_events_per_tick))
        sum(!is.na(d$n_events_per_tick) & d$n_events_per_tick >= 0L)
      else 0L
    } else 0L
    stat_card(n, "Ticks completed", "#dbeafe")
  })

  output$box_events <- renderUI({
    d <- current_data()
    n <- if (!is.null(d) && !is.null(d$sequence_rows))
           length(d$sequence_rows)
         else if (!is.null(d) && !is.null(d$sequence))
           nrow(d$sequence)
         else 0L
    stat_card(n, "Events generated", "#dcfce7")
  })

  output$box_ivs <- renderUI({
    d <- current_data()
    n <- if (!is.null(d) && !is.null(d$intervention_log)) {
      log <- d$intervention_log
      if (is.data.frame(log)) nrow(log)
      else if (is.list(log))  length(log)
      else 0L
    } else 0L
    stat_card(n, "Interventions fired", "#fef9c3")
  })

  # ---- Event feed ---------------------------------------------------
  output$event_feed <- renderText({
    d <- current_data()
    if (is.null(d)) return("Waiting for simulation to start...")

    rows <- if (!is.null(d$sequence_rows)) {
      rr <- Filter(Negate(is.null), d$sequence_rows)
      do.call(rbind, rr)
    } else if (!is.null(d$sequence) && nrow(d$sequence) > 0L) {
      d$sequence
    } else {
      return("No events yet.")
    }

    if (is.null(rows) || nrow(rows) == 0L) return("No events yet.")
    n_show <- min(60L, nrow(rows))
    rows   <- utils::tail(rows, n_show)
    lines  <- paste0(
      sprintf("t=%04d", rows$time), "  Country ", rows$sender,
      "  →  Country ", rows$receiver
    )
    paste(rev(lines), collapse = "\n")
  })

  # ---- Activation heatmap -------------------------------------------
  output$activation_plot <- renderPlot({
    d <- current_data()
    if (is.null(d) || is.null(d$sender_probs_list) ||
        length(d$sender_probs_list) == 0L) {
      plot.new()
      text(0.5, 0.5, "Activation data not yet available.", cex = 1.2)
      return(invisible())
    }

    probs_list <- Filter(Negate(is.null), d$sender_probs_list)
    if (length(probs_list) == 0L) {
      plot.new(); text(0.5, 0.5, "No data yet."); return(invisible())
    }

    # Build matrix: rows = ticks, cols = nodes
    mat <- do.call(rbind, lapply(probs_list, function(p) {
      if (is.null(p)) rep(NA_real_, length(ALL_NODES)) else p
    }))
    # Select top 15 nodes by mean activation
    col_means <- colMeans(mat, na.rm = TRUE)
    top_idx   <- order(col_means, decreasing = TRUE)[seq_len(min(15L, ncol(mat)))]
    mat_sub   <- mat[, top_idx, drop = FALSE]
    node_ids  <- ALL_NODES[top_idx]

    df <- as.data.frame(mat_sub)
    colnames(df) <- paste0("C", node_ids)
    df$tick <- seq_len(nrow(df))

    long <- tidyr::pivot_longer(df, -tick,
                                names_to  = "country",
                                values_to = "p_i")
    long$country <- factor(long$country,
                           levels = paste0("C", node_ids))

    ggplot(long, aes(x = tick, y = country, fill = p_i)) +
      geom_tile() +
      scale_fill_distiller(palette = "YlOrRd", direction = 1,
                           name = expression(p[i](t)),
                           na.value = "grey90") +
      labs(title = "Sender activation probability over ticks",
           x = "Tick", y = NULL) +
      theme_minimal(base_size = 11) +
      theme(axis.text.y = element_text(size = 9))
  })

  # ---- Coefficient forest plot (post-run only) ----------------------
  output$coef_plot <- renderPlot({
    result <- sim_result_rv()
    if (is.null(result) || is.null(result$sequence) ||
        nrow(result$sequence) < 100L) {
      plot.new()
      text(0.5, 0.5,
           "Run the simulation to completion to see coefficient comparison.",
           cex = 1.1)
      return(invisible())
    }

    # Fit REM on generated sequence
    n_nodes <- length(unique(c(result$sequence$sender,
                               result$sequence$receiver,
                               ALL_NODES)))
    rem_sim <- tryCatch(
      fit_rem(result$sequence, REM_FORMULA,
              n = n_nodes, n_nonevents = 200L, base_riskset = BASE_RISKSET),
      error = function(e) NULL
    )
    if (is.null(rem_sim)) {
      plot.new()
      text(0.5, 0.5, paste("REM fit failed (too few events?)"), cex = 1.1)
      return(invisible())
    }

    coef_df <- rbind(
      data.frame(source   = "Empirical",
                 term     = STAT_TERMS,
                 estimate = EMPIRICAL_COEF[STAT_TERMS],
                 row.names = NULL),
      data.frame(source   = "Simulated",
                 term     = STAT_TERMS,
                 estimate = coef(rem_sim)[STAT_TERMS],
                 row.names = NULL)
    )
    coef_df$term <- factor(coef_df$term, levels = rev(STAT_TERMS))

    ggplot(coef_df, aes(x = estimate, y = term, colour = source, shape = source)) +
      geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
      geom_point(size = 4,
                 position = ggplot2::position_dodge(width = 0.4)) +
      scale_colour_manual(values = c(Empirical = "#0d6efd", Simulated = "#dc3545")) +
      labs(title  = "REM coefficients: empirical vs simulated",
           x      = "Coefficient estimate",
           y      = NULL,
           colour = NULL, shape = NULL) +
      theme_minimal(base_size = 12) +
      theme(legend.position = "bottom",
            panel.grid.minor = element_blank())
  })

  # ---- RMSE table ---------------------------------------------------
  output$rmse_table <- renderTable({
    result <- sim_result_rv()
    if (is.null(result) || is.null(result$sequence) ||
        nrow(result$sequence) < 100L)
      return(NULL)

    n_nodes <- length(unique(c(result$sequence$sender,
                               result$sequence$receiver,
                               ALL_NODES)))
    rem_sim <- tryCatch(
      fit_rem(result$sequence, REM_FORMULA,
              n = n_nodes, n_nonevents = 200L, base_riskset = BASE_RISKSET),
      error = function(e) NULL
    )
    if (is.null(rem_sim)) return(NULL)

    emp <- EMPIRICAL_COEF[STAT_TERMS]
    sim <- coef(rem_sim)[STAT_TERMS]

    data.frame(
      Term       = STAT_TERMS,
      Empirical  = round(emp, 3),
      Simulated  = round(sim, 3),
      Difference = round(sim - emp, 3),
      row.names  = NULL,
      check.names = FALSE
    )
  }, striped = TRUE, hover = TRUE, bordered = TRUE)

  # ---- Intervention log table ---------------------------------------
  output$iv_log_table <- renderTable({
    d <- current_data()
    if (is.null(d)) return(NULL)
    log <- d$intervention_log
    if (is.null(log)) return(NULL)
    if (is.list(log) && !is.data.frame(log)) {
      log <- do.call(rbind, Filter(Negate(is.null), log))
    }
    if (is.null(log) || nrow(log) == 0L) {
      return(data.frame(Message = "No interventions have fired yet."))
    }
    log[order(log$tick), ]
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
}

shinyApp(ui, server)
