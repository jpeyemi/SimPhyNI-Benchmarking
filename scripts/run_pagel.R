#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(ape)
  library(phytools)
  library(future)
  library(future.apply)
  library(data.table)
})

# ─────────────────────────────
# Parse arguments
# ─────────────────────────────
parser <- ArgumentParser(description = "Run Pagel correlation analysis on labeled pairs")
parser$add_argument("--tree",        required = TRUE, help = "Input tree in Newick format")
parser$add_argument("--traits",      required = TRUE, help = "CSV file of binary traits")
parser$add_argument("--pair_labels", required = TRUE, help = "pair_labels.csv (trait1, trait2, direction)")
parser$add_argument("--outfile",     required = TRUE, help = "Final output CSV file path")
args <- parser$parse_args()

# ─────────────────────────────
# Setup output directory
# ─────────────────────────────
output_dir <- dirname(args$outfile)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ─────────────────────────────
# Load data
# ─────────────────────────────
cat("Reading tree and traits...\n")
tree        <- read.tree(args$tree)
traits      <- as.matrix(read.csv(args$traits, row.names = 1))
pair_labels <- read.csv(args$pair_labels, stringsAsFactors = FALSE)

if (!all(rownames(traits) %in% tree$tip.label))
  stop("Mismatch between trait row names and tree tip labels")

cat("Loaded", ncol(traits), "traits for", length(tree$tip.label), "taxa.\n")

# ─────────────────────────────
# Pre-filter degenerate traits
# ─────────────────────────────
trait_variance <- colSums(traits) > 0 & colSums(traits) < nrow(traits)
n_before <- nrow(pair_labels)
pair_labels <- pair_labels[
  pair_labels$trait1 %in% colnames(traits)[trait_variance] &
  pair_labels$trait2 %in% colnames(traits)[trait_variance], ]
n_dropped <- n_before - nrow(pair_labels)
if (n_dropped > 0)
  cat("Dropped", n_dropped, "pairs with degenerate (constant) traits.\n")

cat("Running Pagel on", nrow(pair_labels), "labeled pairs.\n")

# ─────────────────────────────
# Parallel setup
# ─────────────────────────────
num_cores <- min(16, parallel::detectCores(logical = FALSE) - 1)
cat("Using", num_cores, "cores.\n")

# Use multicore (fork) on Unix for shared memory — avoids serializing traits/tree
if (.Platform$OS.type == "unix") {
  plan(multicore, workers = num_cores)
} else {
  plan(multisession, workers = num_cores)
  options(future.globals.maxSize = 2000 * 1024^2)  # 2 GB (needed for multisession only)
}

# ─────────────────────────────
# Helper: analyze one pair (receives pre-sliced vectors)
# ─────────────────────────────
analyze_pair <- function(x, y, trait1, trait2, label, tree) {
  res <- tryCatch(
    fitPagel(tree, x, y, method = "fitMk", model = "ER", dep.var = "xy"),
    error = function(e) NULL
  )
  if (is.null(res)) {
    return(data.frame(Trait1 = trait1, Trait2 = trait2,
                      P_Value = NA, Direction = NA, label = label))
  }

  p_value <- res$P[1]
  Q <- res$dependent.Q
  direction <- if ((Q[4, 2] + Q[4, 3]) > (Q[2, 4] + Q[3, 4])) -1 else 1

  data.frame(Trait1 = trait1, Trait2 = trait2,
             P_Value = p_value, Direction = direction, label = label)
}

# ─────────────────────────────
# Chunked execution with checkpointing
# ─────────────────────────────
chunk_size <- 100
n_pairs    <- nrow(pair_labels)
chunks     <- split(seq_len(n_pairs), ceiling(seq_len(n_pairs) / chunk_size))

temp_dir <- file.path(output_dir, "pagel_chunks")
if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE)

# Pre-compute row names once — constant across all pairs
rn <- rownames(traits)

cat("Starting Pagel analysis (resumable)...\n")
start_time       <- Sys.time()
completed_chunks <- 0

for (chunk_id in seq_along(chunks)) {
  chunk_file <- file.path(temp_dir, sprintf("chunk_%03d.csv", chunk_id))

  if (file.exists(chunk_file)) {
    cat("Skipping chunk", chunk_id, "(already completed)\n")
    completed_chunks <- completed_chunks + 1
    next
  }

  chunk_start <- Sys.time()
  cat("Processing chunk", chunk_id, "of", length(chunks), "\n")

  idx_range <- chunks[[chunk_id]]
  chunk_df  <- pair_labels[idx_range, ]

  # Pre-slice trait vectors before dispatch — workers receive small named vectors
  # instead of the full traits matrix, eliminating per-future serialization overhead
  xs     <- lapply(chunk_df$trait1, function(t) { v <- traits[, t]; names(v) <- rn; v })
  ys     <- lapply(chunk_df$trait2, function(t) { v <- traits[, t]; names(v) <- rn; v })

  chunk_results <- future_mapply(
    analyze_pair,
    x      = xs,
    y      = ys,
    trait1 = chunk_df$trait1,
    trait2 = chunk_df$trait2,
    label  = chunk_df$direction,
    MoreArgs = list(tree = tree),
    SIMPLIFY = FALSE,
    future.seed = NULL
  )

  chunk_out <- rbindlist(chunk_results)
  fwrite(chunk_out, chunk_file)

  completed_chunks <- completed_chunks + 1
  elapsed    <- difftime(Sys.time(), start_time, units = "mins")
  chunk_time <- difftime(Sys.time(), chunk_start, units = "secs")
  avg_per    <- as.numeric(elapsed) / completed_chunks
  eta_mins   <- avg_per * (length(chunks) - completed_chunks)
  cat(sprintf("Chunk %d done in %.1fs | %d/%d (%.1f%%) | ETA: %.1f min\n",
              chunk_id, chunk_time, completed_chunks, length(chunks),
              100 * completed_chunks / length(chunks), eta_mins))
}

# ─────────────────────────────
# Merge results
# ─────────────────────────────
cat("Combining chunk files...\n")
chunk_files <- sort(list.files(temp_dir, pattern = "chunk_.*\\.csv$", full.names = TRUE))
all_results <- rbindlist(lapply(chunk_files, fread))

fwrite(all_results, args$outfile)
cat("Analysis complete. Results written to", args$outfile, "\n")
