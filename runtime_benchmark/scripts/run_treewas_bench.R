#!/usr/bin/env Rscript
# TreeWAS runtime benchmark — all-vs-all, no pair_labels, no accuracy evaluation.

suppressPackageStartupMessages({
  library(devtools)
  if (!requireNamespace("treeWAS", quietly = TRUE))
    install_github("caitiecollins/treeWAS", build_vignettes = FALSE)
  library(treeWAS)
  library(ape)
  library(parallel)
  library(argparse)
})

parser <- ArgumentParser(description = "TreeWAS all-vs-all runtime benchmark")
parser$add_argument("--tree",    required = TRUE, help = "Newick tree file")
parser$add_argument("--traits",  required = TRUE, help = "Binary traits CSV (samples x traits)")
parser$add_argument("--threads", type = "integer", default = 16)
parser$add_argument("--outdir",  required = TRUE, help = "Output directory")
args <- parser$parse_args()

dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

tree      <- read.tree(args$tree)
gene_data <- as.matrix(read.csv(args$traits, row.names = 1))

num_traits    <- ncol(gene_data)
gene_colnames <- colnames(gene_data)
gene_rownames <- rownames(gene_data)

# Pre-filter to binary traits
binary_mask    <- apply(gene_data, 2, function(col) length(unique(col[!is.na(col)])) == 2)
binary_indices <- which(binary_mask)
message(paste0("Binary traits: ", length(binary_indices), " / ", num_traits))

# Core count: honour SLURM allocation or --threads flag
n_cores <- min(
  args$threads,
  as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = as.character(args$threads))),
  max(1L, detectCores() - 1L),
  length(binary_indices)
)
message(paste0("Running on ", n_cores, " core(s)"))

process_phenotype <- function(pheno_idx) {
  pheno_name <- gene_colnames[pheno_idx]
  phen       <- gene_data[, pheno_idx]
  names(phen) <- gene_rownames

  tryCatch({
    treeWAS(
      gene_data,
      phen,
      tree               = tree,
      plot.tree          = FALSE,
      plot.manhattan     = FALSE,
      plot.null.dist     = FALSE,
      phen.type          = "discrete",
      p.value.correct    = FALSE,
      p.value            = 1,
    )
    1L  # success
  }, error = function(e) {
    message(paste0("Error at phenotype ", pheno_name, ": ", e$message))
    0L
  })
}

result_list <- mclapply(binary_indices, process_phenotype, mc.cores = n_cores)
n_done <- sum(unlist(result_list))

writeLines(as.character(n_done), file.path(args$outdir, "n_completed.txt"))
message(paste0("TreeWAS complete: ", n_done, "/", length(binary_indices), " phenotypes processed."))
