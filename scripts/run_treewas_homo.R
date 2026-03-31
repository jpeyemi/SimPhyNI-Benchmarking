## Install and load necessary packages
if (!require(devtools)) install.packages("devtools", dependencies = TRUE)
library(devtools)
if (!require(treeWAS)) install_github("caitiecollins/treeWAS", build_vignettes = FALSE)
library(treeWAS)
library(dplyr)
library(tidyr)
library(ggplot2)
if (!requireNamespace("argparse", quietly = TRUE)) install.packages("argparse")
library(argparse)
library(parallel)

# Argument parsing
parser <- ArgumentParser(description = "Run TreeWAS analysis (GWAS mode: all traits as phenotypes)")
parser$add_argument("--tree",        type = "character", required = TRUE,  help = "Input tree file")
parser$add_argument("--traits",      type = "character", required = TRUE,  help = "Input traits pivot table file")
parser$add_argument("--pair_labels", type = "character", required = TRUE,  help = "pair_labels.csv (trait1, trait2, direction)")
parser$add_argument("--terminal",    type = "character", required = TRUE,  help = "Output file for terminal analysis")
parser$add_argument("--simultaneous",type = "character", required = TRUE,  help = "Output file for simultaneous analysis")
parser$add_argument("--subsequent",  type = "character", required = TRUE,  help = "Output file for subsequent analysis")
args <- parser$parse_args()

# Access arguments
tree_file        <- args$tree
traits_file      <- args$traits
pair_labels_file <- args$pair_labels
terminal_output     <- args$terminal
simultaneous_output <- args$simultaneous
subsequent_output   <- args$subsequent

print(paste("Tree file:", tree_file))
print(paste("Traits file:", traits_file))
print(paste("Pair labels file:", pair_labels_file))

# Read inputs
tree       <- read.tree(file = tree_file)
gene_data  <- as.matrix(read.csv(traits_file, row.names = 1))
pair_labels <- read.csv(pair_labels_file, stringsAsFactors = FALSE)

# Ensure output directories exist
output_dirs <- unique(dirname(c(terminal_output, simultaneous_output, subsequent_output)))
for (dir in output_dirs) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
}

num_traits    <- ncol(gene_data)
gene_colnames <- colnames(gene_data)
gene_rownames <- rownames(gene_data)

# Build pair-label lookup: named vector indexed by "t1|||t2" key (both orderings)
make_key <- function(t1, t2) paste(sort(c(t1, t2)), collapse = "|||")
pair_key_vec <- mapply(make_key, pair_labels$trait1, pair_labels$trait2)
label_lookup <- setNames(pair_labels$direction, pair_key_vec)

# Helper: extract significant hits for a given phenotype column and return results
# filtered to only labeled pairs involving this phenotype.
extract_labeled <- function(analysis_result, pheno_name, all_col_names) {
  if (is.null(analysis_result$sig.snps) ||
      identical(analysis_result$sig.snps, "No significant SNPs found.")) return(NULL)
  sig <- analysis_result$sig.snps
  if (nrow(sig) == 0) return(NULL)

  geno_names <- rownames(sig)
  rows <- lapply(geno_names, function(geno) {
    key   <- make_key(pheno_name, geno)
    label <- label_lookup[key]
    if (is.na(label)) return(NULL)  # pair not in pair_labels; skip
    data.frame(
      phenotype = pheno_name,
      genotype  = geno,
      Score     = sig[geno, "score"],
      p_value   = sig[geno, "p.value"],
      label     = label,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, Filter(Negate(is.null), rows))
}

# Process one phenotype: run treeWAS with this trait as phenotype,
# all other traits as the gene matrix (background for null distribution).
process_phenotype <- function(pheno_idx) {
  pheno_name <- gene_colnames[pheno_idx]
  phen       <- gene_data[, pheno_idx]
  names(phen) <- gene_rownames

  # Skip if not binary
  if (length(unique(phen)) != 2) return(NULL)

  # Use all traits as gene matrix (background); treeWAS handles the pheno internally
  out_list <- tryCatch({
    result <- treeWAS(
      gene_data,
      phen,
      tree = tree,
      plot.tree         = FALSE,
      plot.manhattan    = FALSE,
      plot.null.dist    = FALSE,
      phen.type         = "discrete",
      p.value.correct   = FALSE,
      p.value           = 1,
    )

    list(
      terminal     = extract_labeled(result$terminal,     pheno_name, gene_colnames),
      simultaneous = extract_labeled(result$simultaneous, pheno_name, gene_colnames),
      subsequent   = extract_labeled(result$subsequent,   pheno_name, gene_colnames)
    )
  }, error = function(e) {
    message(paste0("Error at phenotype ", pheno_name, ": ", e$message))
    NULL
  })

  return(out_list)
}

# Run all phenotypes in parallel
all_indices <- seq_len(num_traits)
result_list <- mclapply(all_indices, process_phenotype, mc.cores = 4)

# Combine results across all phenotypes
combined_terminal     <- bind_rows(lapply(result_list, function(x) if (!is.null(x)) x$terminal))
combined_simultaneous <- bind_rows(lapply(result_list, function(x) if (!is.null(x)) x$simultaneous))
combined_subsequent   <- bind_rows(lapply(result_list, function(x) if (!is.null(x)) x$subsequent))

# Write output files
write.csv(combined_terminal,     terminal_output,     row.names = FALSE)
write.csv(combined_simultaneous, simultaneous_output, row.names = FALSE)
write.csv(combined_subsequent,   subsequent_output,   row.names = FALSE)
