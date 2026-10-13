# Run this once to install all Shiny app dependencies.
pkgs <- c(
  "shiny",        # >= 1.8.0 (required for ExtendedTask)
  "bslib",
  "shinyWidgets",
  "future",
  "promises",
  "ggplot2",
  "tidyr",
  "survival",
  "devtools"
)
install.packages(pkgs[!pkgs %in% rownames(installed.packages())])
