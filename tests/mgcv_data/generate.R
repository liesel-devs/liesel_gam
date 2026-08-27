#!/usr/bin/env Rscript

# Regenerate the committed Columbus example data used by Python-only tests.
# This script is intentionally adjacent to the generated CSV assets.

args <- commandArgs(trailingOnly = TRUE)
output <- if (length(args)) args[[1]] else dirname(normalizePath(sys.frame(1)$ofile))
dir.create(output, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(mgcv))
data(columb, package = "mgcv")
data(columb.polys, package = "mgcv")

write.csv(columb, file.path(output, "columb.csv"), row.names = TRUE)

polygon_rows <- lapply(seq_along(columb.polys), function(index) {
  polygon <- columb.polys[[index]]
  data.frame(
    label = names(columb.polys)[[index]],
    vertex = seq_len(nrow(polygon)),
    x = polygon[, 1],
    y = polygon[, 2]
  )
})
write.csv(
  do.call(rbind, polygon_rows),
  file.path(output, "columb_polys.csv"),
  row.names = FALSE
)
