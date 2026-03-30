#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(argparse)
  library(ape)
  library(phytools)
  library(future)
  library(future.apply)
})

# ─────────────────────────────
# Parse arguments
# ─────────────────────────────
parser <- ArgumentParser(description = "Run Pagel correlation analysis (fast + resumable)")
parser$add_argument("--tree", required = TRUE, help = "Input tree in Newick format")
parser$add_argument("--traits", required = TRUE, help = "CSV file of binary traits")
parser$add_argument("--outfile", required = TRUE, help = "Final output CSV file path")
args <- parser$parse_args()

nwk_file <- args$tree
traits_file <- args$traits
out <- args$outfile

# ─────────────────────────────
# Setup output directory
# ─────────────────────────────
output_dir <- dirname(out)
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# ─────────────────────────────
# Load data
# ─────────────────────────────
cat("📖 Reading tree and traits...\n")
tree <- read.tree(nwk_file)
traits <- as.matrix(read.csv(traits_file, row.names = 1))  # matrix = faster access

if (!all(rownames(traits) %in% tree$tip.label))
  stop("❌ Mismatch between trait row names and tree tip labels")

num_traits <- ncol(traits)
cat("✅ Loaded", num_traits, "traits for", length(tree$tip.label), "taxa.\n")

# ─────────────────────────────
# Parallel setup
# ─────────────────────────────
num_cores <- 16
cat("⚙️  Using", num_cores, "cores.\n")

plan(multisession, workers = num_cores)

# ─────────────────────────────
# Helper: analyze one pair
# ─────────────────────────────
analyze_pair <- function(i, traits, tree) {
  if (i + 1 > ncol(traits)) return(NULL)
  
  trait1 <- colnames(traits)[i]
  trait2 <- colnames(traits)[i + 1]
  x <- traits[, i]
  y <- traits[, i + 1]
  names(x) <- names(y) <- rownames(traits)
  
  res <- tryCatch(
    fitPagel(tree, x, y, method = "fitMk", model = "ARD", dep.var = "xy"),  # faster than ARD
    error = function(e) NULL
  )
  if (is.null(res)) {
    return(data.frame(Trait1 = trait1, Trait2 = trait2,
                      P_Value = NA, Direction = NA))
  }
  
  p_value <- res$P[1]
  Q <- res$dependent.Q
  direction <- if ((Q[4, 2] + Q[4, 3]) > (Q[2, 4] + Q[3, 4])) -1 else 1
  
  data.frame(Trait1 = trait1, Trait2 = trait2,
             P_Value = p_value, Direction = direction)
}

# ─────────────────────────────
# Chunked execution with checkpointing
# ─────────────────────────────
indices <- seq(1, num_traits, by = 2)
chunk_size <- 100  # larger chunks = fewer I/O operations
chunks <- split(indices, ceiling(seq_along(indices) / chunk_size))

temp_dir <- file.path(output_dir, "pagel_chunks")
if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE)

cat("🚀 Starting Pagel analysis (fast + resumable)...\n")

for (chunk_id in seq_along(chunks)) {
  chunk_file <- file.path(temp_dir, sprintf("chunk_%03d.csv", chunk_id))
  
  if (file.exists(chunk_file)) {
    cat("⏩ Skipping chunk", chunk_id, "(already completed)\n")
    next
  }
  
  cat("▶️  Processing chunk", chunk_id, "of", length(chunks), "\n")
  
  chunk <- chunks[[chunk_id]]
  chunk_results <- future_lapply(chunk, function(i)
    analyze_pair(i, traits, tree),
    future.seed = TRUE
  )
  
  chunk_df <- do.call(rbind, chunk_results)
  write.table(chunk_df, chunk_file, sep = ",", row.names = FALSE, col.names = TRUE)
}

# ─────────────────────────────
# Merge results after all chunks done
# ─────────────────────────────
cat("🧩 Combining all chunk files into final output...\n")
chunk_files <- sort(list.files(temp_dir, pattern = "chunk_.*\\.csv$", full.names = TRUE))
all_results <- do.call(rbind, lapply(chunk_files, read.csv))

write.csv(all_results, out, row.names = FALSE)
cat("✅ Analysis complete. Final results written to", out, "\n")
