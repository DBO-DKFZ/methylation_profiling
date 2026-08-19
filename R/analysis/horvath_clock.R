library(methylclock)

source("lib/utils.R")

dir.create(dirname(path("horvath_ages_v1")), recursive = TRUE, showWarnings = FALSE)

compute_horvath_age <- function(betas_path, output_path, label = "") {
  betas <- load_betas(betas_path)
  meta <- get_meta()

  sample_ids <- colnames(betas)
  age_vector <- meta$approxAge[match(sample_ids, as.character(meta$slideId))]
  cat("Matched ages for", sum(!is.na(age_vector)), "of", length(age_vector), "samples\n")

  cat("Running Horvath clock for", label, "...\n")
  ages <- DNAmAge(betas, clocks = "skinHorvath", age = age_vector, cell.count = FALSE)

  cat("Writing results to", output_path, "\n")
  fwrite(ages, output_path)
  cat("Done.\n\n")
}

compute_horvath_age(path("betas_v1"), path("horvath_ages_v1"), label = "EPICv1")
compute_horvath_age(path("betas_v2"), path("horvath_ages_v2"), label = "EPICv2")
