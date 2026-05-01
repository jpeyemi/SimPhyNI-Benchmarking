#!/usr/bin/env Rscript
# Pagel runtime benchmark — all-vs-all, no pair_labels, no accuracy evaluation.

suppressPackageStartupMessages({
  library(argparse)
  library(ape)
  library(phytools)
  library(future)
  library(future.apply)
  library(data.table)
})

parser <- ArgumentParser(description = "Pagel all-vs-all runtime benchmark")
parser$add_argument("--tree",    required = TRUE, help = "Newick tree file")
parser$add_argument("--traits",  required = TRUE, help = "Binary traits CSV (samples x traits)")
parser$add_argument("--threads", type = "integer", default = 16)
parser$add_argument("--outfile", required = TRUE, help = "Output CSV path")
args <- parser$parse_args()

output_dir <- dirname(args$outfile)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cat("Reading tree and traits...\n")
tree   <- read.tree(args$tree)
traits <- as.matrix(read.csv(args$traits, row.names = 1))

if (!all(rownames(traits) %in% tree$tip.label))
  stop("Mismatch between trait row names and tree tip labels")
cat("Loaded", ncol(traits), "traits for", length(tree$tip.label), "taxa.\n")

# Pre-filter degenerate traits (constant columns)
valid_mask <- colSums(traits) > 0 & colSums(traits) < nrow(traits)
valid_traits <- colnames(traits)[valid_mask]
cat("Valid (non-constant) traits:", length(valid_traits), "\n")

# Build all-vs-all pair list (upper triangle only)
pairs <- combn(valid_traits, 2, simplify = FALSE)
cat("Running Pagel on", length(pairs), "all-vs-all pairs.\n")

# Parallel setup
num_cores <- min(args$threads, parallel::detectCores(logical = FALSE) - 1)
cat("Using", num_cores, "cores.\n")

if (.Platform$OS.type == "unix") {
  plan(multicore, workers = num_cores)
} else {
  plan(multisession, workers = num_cores)
  options(future.globals.maxSize = 2000 * 1024^2)
}

analyze_pair <- function(pair, traits, tree) {
  trait1 <- pair[1]; trait2 <- pair[2]
  x <- traits[, trait1]; names(x) <- rownames(traits)
  y <- traits[, trait2]; names(y) <- rownames(traits)
  res <- tryCatch(
    fitPagel(tree, x, y, method = "fitMk", model = "ARD", dep.var = "xy"),
    error = function(e) NULL
  )
  if (is.null(res))
    return(data.frame(Trait1 = trait1, Trait2 = trait2, P_Value = NA_real_, Direction = NA_integer_))
  Q <- res$dependent.Q
  direction <- if ((Q[4, 2] + Q[4, 3]) > (Q[2, 4] + Q[3, 4])) -1L else 1L
  data.frame(Trait1 = trait1, Trait2 = trait2, P_Value = res$P[1], Direction = direction)
}

# Chunked execution with checkpointing (resumable)
chunk_size <- 100
chunks     <- split(seq_along(pairs), ceiling(seq_along(pairs) / chunk_size))
temp_dir   <- file.path(output_dir, "pagel_chunks")
dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)

start_time       <- Sys.time()
completed_chunks <- 0

for (chunk_id in seq_along(chunks)) {
  chunk_file <- file.path(temp_dir, sprintf("chunk_%04d.csv", chunk_id))
  if (file.exists(chunk_file)) {
    completed_chunks <- completed_chunks + 1
    next
  }
  chunk_start  <- Sys.time()
  chunk_pairs  <- pairs[chunks[[chunk_id]]]

  chunk_results <- future_lapply(
    chunk_pairs,
    analyze_pair,
    traits     = traits,
    tree       = tree,
    future.seed = NULL
  )

  chunk_out <- rbindlist(chunk_results)
  fwrite(chunk_out, chunk_file)

  completed_chunks <- completed_chunks + 1
  elapsed  <- difftime(Sys.time(), start_time, units = "mins")
  avg_per  <- as.numeric(elapsed) / completed_chunks
  eta_mins <- avg_per * (length(chunks) - completed_chunks)
  cat(sprintf("Chunk %d/%d done in %.1fs | ETA: %.1f min\n",
              chunk_id, length(chunks),
              difftime(Sys.time(), chunk_start, units = "secs"),
              eta_mins))
}

cat("Combining chunks...\n")
chunk_files <- sort(list.files(temp_dir, pattern = "chunk_.*\\.csv$", full.names = TRUE))
all_results <- rbindlist(lapply(chunk_files, fread))
fwrite(all_results, args$outfile)
cat("Done. Results written to", args$outfile, "\n")
