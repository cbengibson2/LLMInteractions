# =============================================================================
# llm_rem_compare_conditions.R
# Condition comparison: predict_obs vs. multi-agent (REM sender + LLM receiver)
#
# Loads results from both simulation studies and produces comparison plots:
#
#   1. RMSE by domain × condition × provider (bar chart)
#      — "how much does the REM sender mechanism improve overall accuracy?"
#
#   2. Bias heatmap by condition
#      — same term × domain layout as per-study plots, but faceted by condition
#        so you can see which biases are reduced and which persist
#
#   3. Per-term bias reduction dot plot
#      — signed bias for each term averaged across domains, condition on y-axis
#        so systematic gains/losses per term are immediately visible
#
# Requires:
#   multiagent_multidomain_results.RData
#   predict_multidomain_results.RData
# =============================================================================

library(ggplot2)
library(llrem)

STUDY_DIR <- Sys.getenv("STUDY_DIR", ".")  # project root

# =============================================================================
# 1. LOAD
# =============================================================================

ma  <- new.env(parent = emptyenv())
prd <- new.env(parent = emptyenv())

load(file.path(STUDY_DIR, "multiagent_multidomain_results.RData"), envir = ma)
load(file.path(STUDY_DIR, "predict_multidomain_results.RData"),    envir = prd)

ACTIVE_DOMAINS   <- c("War", "Classroom", "Gangs", "Email", "ftf",
                      "PMs", "SMS", "Twitter", "Phone")
ACTIVE_PROVIDERS <- c("bedrock_claude", "bedrock_llama")

stat_terms <- c("SndInitiation", "IOviolation", "Repetition", "RecRecencyRank",
                "NOTPSnd", "OutDegSnd", "WeightedRec", "Imitation", "NITPSnd")

provider_labels <- c(
  bedrock_claude = "Bedrock/Claude (Haiku 4.5)",
  bedrock_llama  = "Bedrock/Llama (3.1 70B)"
)

condition_labels <- c(
  predict    = "Predict (baseline)",
  multiagent = "Multi-agent (REM sender)"
)

provider_palette <- c(
  "Bedrock/Claude (Haiku 4.5)" = "#1B7837",
  "Bedrock/Llama (3.1 70B)"   = "#762A83"
)

condition_palette <- c(
  "Predict (baseline)"        = "#D6604D",
  "Multi-agent (REM sender)"  = "#4393C3"
)

# =============================================================================
# 2. COMBINE RMSE TABLES
# =============================================================================

ma$rmse_domain$condition  <- "Multi-agent (REM sender)"
prd$rmse_domain$condition <- "Predict (baseline)"

rmse_all <- rbind(ma$rmse_domain, prd$rmse_domain)
rmse_all$domain    <- factor(rmse_all$domain,    levels = ACTIVE_DOMAINS)
rmse_all$condition <- factor(rmse_all$condition,
                              levels = c("Predict (baseline)",
                                         "Multi-agent (REM sender)"))

cat("RMSE comparison (domain averages across networks):\n")
print(rmse_all[order(rmse_all$domain, rmse_all$source, rmse_all$condition), ],
      row.names = FALSE)

# =============================================================================
# 3. COMBINE BIAS TABLES
# =============================================================================

ma$bias_df$condition  <- "Multi-agent (REM sender)"
prd$bias_df$condition <- "Predict (baseline)"

bias_all <- rbind(ma$bias_df, prd$bias_df)
bias_all$term      <- factor(bias_all$term,      levels = rev(stat_terms))
bias_all$domain    <- factor(bias_all$domain,    levels = ACTIVE_DOMAINS)
bias_all$condition <- factor(bias_all$condition,
                              levels = c("Predict (baseline)",
                                         "Multi-agent (REM sender)"))

# =============================================================================
# 4. PLOT 1: RMSE BAR CHART — condition × domain × provider
# =============================================================================

p_rmse <- ggplot(rmse_all,
                  aes(x = domain, y = RMSE,
                      fill = condition,
                      alpha = source)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.35) +
  scale_fill_manual(values = condition_palette) +
  scale_alpha_manual(
    values = c("Bedrock/Claude (Haiku 4.5)" = 1.0,
               "Bedrock/Llama (3.1 70B)"   = 0.55),
    name = "Provider"
  ) +
  labs(
    x       = NULL,
    y       = "RMSE vs empirical sufficient statistic means",
    fill    = "Condition",
    title   = "RMSE by domain and condition",
    subtitle = "Solid = Claude (Haiku 4.5); transparent = Llama (3.1 70B)",
    caption = "RMSE averaged across networks within each domain."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x        = element_text(angle = 35, hjust = 1),
    legend.position    = "bottom",
    panel.grid.major.x = element_blank()
  )

print(p_rmse)
ggsave(file.path(STUDY_DIR, "compare_rmse.pdf"), p_rmse, width = 10, height = 5)

# =============================================================================
# 5. PLOT 2: BIAS HEATMAP — faceted by condition × provider
# =============================================================================

bias_limit <- max(abs(bias_all$bias), na.rm = TRUE)

p_bias <- ggplot(bias_all, aes(x = domain, y = term, fill = bias)) +
  geom_tile(colour = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%+.2f", bias)),
            size = 2.4, colour = "grey20") +
  scale_fill_gradient2(
    low      = "#D6604D",
    mid      = "white",
    high     = "#4393C3",
    midpoint = 0,
    limits   = c(-bias_limit, bias_limit),
    name     = "LLM − empirical"
  ) +
  facet_grid(source ~ condition) +
  labs(
    x        = NULL,
    y        = NULL,
    title    = "LLM bias by domain, term, and condition",
    subtitle = "Fill = mean(LLM) − mean(empirical); red = under, blue = over",
    caption  = "Domain averages across networks."
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x      = element_text(angle = 35, hjust = 1),
    strip.text       = element_text(size = 8, face = "bold"),
    strip.background = element_rect(fill = "grey92", colour = "grey60", linewidth = 0.5),
    panel.border     = element_rect(colour = "grey60", fill = NA, linewidth = 0.5),
    panel.grid       = element_blank(),
    legend.position  = "right"
  )

print(p_bias)
ggsave(file.path(STUDY_DIR, "compare_bias_heatmap.pdf"), p_bias, width = 13, height = 7)

# =============================================================================
# 6. PLOT 3: PER-TERM BIAS REDUCTION — domain-averaged bias by term × condition
# =============================================================================
# Collapses across domains to show which terms improve, worsen, or are
# unaffected by the REM sender mechanism.

term_bias <- aggregate(bias ~ term + condition + source + provider,
                        data = bias_all, FUN = mean, na.rm = TRUE)
term_bias$term      <- factor(term_bias$term,      levels = rev(stat_terms))
term_bias$condition <- factor(term_bias$condition,
                               levels = c("Predict (baseline)",
                                          "Multi-agent (REM sender)"))

p_term <- ggplot(term_bias, aes(x = bias, y = term,
                                 colour = condition, shape = source)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
  geom_point(size = 3.5, alpha = 0.85,
             position = position_dodge(width = 0.5)) +
  scale_colour_manual(values = condition_palette) +
  scale_shape_manual(
    values = c("Bedrock/Claude (Haiku 4.5)" = 16L,
               "Bedrock/Llama (3.1 70B)"   = 17L)
  ) +
  labs(
    x        = "Mean bias (LLM − empirical), averaged across domains",
    y        = NULL,
    colour   = "Condition",
    shape    = "Provider",
    title    = "Per-term bias by condition",
    subtitle = "Left of zero = LLM under-expresses; right = over-expresses",
    caption  = "Each point is the mean signed bias across all 9 domains."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position    = "bottom",
    panel.grid.major.y = element_blank(),
    panel.grid.minor   = element_blank()
  )

print(p_term)
ggsave(file.path(STUDY_DIR, "compare_term_bias.pdf"), p_term, width = 9, height = 6)

cat(sprintf("\nPlots saved to %s/compare_*.pdf\n", STUDY_DIR))
