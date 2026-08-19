library(methyLImp2)
library(BiocParallel)

source("lib/utils.R")

impute_methylation <- function(input_file, output_file) {

  cat("Reading data from:", input_file, "\n")
  betas <- fread(input_file, header = TRUE)
  betas <- as.data.frame(betas[, -1], row.names = betas[[1]])

  cpg_names <- rownames(betas)

  # methyLImp2 imputes per chromosome; CpG ids are "chr<n>-<pos>", so the chromosome is the part before the hyphen.
  chr_info <- data.frame(
    cpg = cpg_names,
    chr = sub("-.*", "", cpg_names)
  )
  betas_matrix <- as.matrix(t(betas))

  cat("Performing imputation...\n")
  betas_imputed <- methyLImp2(betas_matrix, 
                              type = 'user', 
                              annotation = chr_info,
                              BPPARAM = MulticoreParam(workers = config$workers))

  cat("Saving imputed data to:", output_file, "\n")
  result <- as.data.table(t(betas_imputed), keep.rownames = TRUE)
  fwrite(result, output_file)
  
  cat("Done!\n\n")
  
  return(invisible(betas_imputed))
}

input_files  <- c(path("betas_test"),         path("betas_train"))
output_files <- c(path("betas_imputed_test"), path("betas_imputed_train"))

for (i in seq_along(input_files)) {
  impute_methylation(input_files[i], output_files[i])
}
