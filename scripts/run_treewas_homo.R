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
parser <- ArgumentParser(description = "Run TreeWAS analysis")
parser$add_argument("--tree", type = "character", required = TRUE, help = "Input tree file")
parser$add_argument("--traits", type = "character", required = TRUE, help = "Input traits pivot table file")
parser$add_argument("--terminal", type = "character", required = TRUE, help = "Output file for terminal analysis")
parser$add_argument("--simultaneous", type = "character", required = TRUE, help = "Output file for simultaneous analysis")
parser$add_argument("--subsequent", type = "character", required = TRUE, help = "Output file for subsequent analysis")
args <- parser$parse_args()

# Access arguments
tree_file <- args$tree
traits_file <- args$traits
terminal_output <- args$terminal
simultaneous_output <- args$simultaneous
subsequent_output <- args$subsequent

print(paste("Tree file:", tree_file))
print(paste("Traits file:", traits_file))
print(paste("Terminal output file:", terminal_output))
print(paste("Simultaneous output file:", simultaneous_output))
print(paste("Subsequent output file:", subsequent_output))

# Read input tree and traits
tree <- read.tree(file = tree_file)
gene_data <- as.matrix(read.csv(traits_file, row.names = 1))

# Ensure output directories exist
output_dirs <- unique(dirname(c(terminal_output, simultaneous_output, subsequent_output)))
for (dir in output_dirs) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
}

# Set number of traits
num_traits <- ncol(gene_data)

# Pre-compute constants shared across all parallel workers
gene_colnames  <- colnames(gene_data)
gene_rownames  <- rownames(gene_data)
fixed_base_idx <- 1:min(300L, num_traits)

# Helper: filter one analysis result to TARGET_SNPS and build output data.frame
extract_sig <- function(analysis_result, col_name, TARGET_SNPS) {
  if (is.null(analysis_result$sig.snps) ||
      "No significant SNPs found." %in% analysis_result$sig.snps) return(NULL)
  sig <- analysis_result$sig.snps[rownames(analysis_result$sig.snps) %in% TARGET_SNPS, , drop = FALSE]
  if (nrow(sig) == 0) return(NULL)
  data.frame(SNP = rownames(sig), Score = sig$score,
             First = col_name, Second = rownames(sig),
             p_value = sig$p.value, stringsAsFactors = FALSE)
}

# Define function for processing one trait
process_trait <- function(i) {
  col_name <- gene_colnames[i + 1] # Trait i+1 is the PHENOTYPE
  phen <- gene_data[, i + 1]
  names(phen) <- gene_rownames

  # Define the two traits whose results we want to keep
  TARGET_SNPS <- gene_colnames[c(i, i + 1)]
  all_indices <- union(fixed_base_idx, c(i, i + 1))

  # Subset the gene matrix
  gene_matrix <- gene_data[, all_indices]
  
  out_list <- tryCatch({
    result <- treeWAS(
      gene_matrix,
      phen,
      tree = tree,
      plot.tree = FALSE,
      plot.manhattan = FALSE,
      plot.null.dist = FALSE,
      phen.type = "discrete",
      p.value.correct = FALSE,
      p.value = 1,
    )
    
    list(
      terminal     = extract_sig(result$terminal,     col_name, TARGET_SNPS),
      simultaneous = extract_sig(result$simultaneous, col_name, TARGET_SNPS),
      subsequent   = extract_sig(result$subsequent,   col_name, TARGET_SNPS)
    )
    
  }, error = function(e) {
    message(paste0("Error at trait ", col_name, ": ", e$message))
    # Return dummy NA rows (simplified error handling)
    return(NULL) 
  })
  
  return(out_list)
}

# Pre-filter to only dispatch workers for trait pairs with a binary phenotype (i+1)
candidate_i <- seq(1, num_traits - 1, by = 2)
is_binary   <- sapply(candidate_i + 1, function(j) length(unique(gene_data[, j])) == 2)
valid_i     <- candidate_i[is_binary]
result_list <- mclapply(valid_i, process_trait, mc.cores = 4)

# Combine the results
combined_results_terminal <- bind_rows(lapply(result_list, `[[`, "terminal"))
combined_results_simultaneous <- bind_rows(lapply(result_list, `[[`, "simultaneous"))
combined_results_subsequent <- bind_rows(lapply(result_list, `[[`, "subsequent"))

# Write output files
write.csv(combined_results_terminal, terminal_output, row.names = FALSE)
write.csv(combined_results_simultaneous, simultaneous_output, row.names = FALSE)
write.csv(combined_results_subsequent, subsequent_output, row.names = FALSE)
