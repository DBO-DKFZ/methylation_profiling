source("lib/utils.R")
source("lib/cnv_helpers.R")

meta <- get_meta()
sample_map <- fread(path("methylation_samples_cleaned"))

# NV reference: all non-test-clinic nevi (excluded from the classifier's external test set).
nevus_slides <- meta[groupedPrimaryDiagnosisPatho == "NV" & clinic != config$test_clinic,as.character(slideId)]
nevus_keys <- sample_map[as.character(slideId) %in% nevus_slides, sentrix_key]
cat("Found", length(nevus_keys), "nevus samples for reference\n")

# Restrict analysis to diagnoses of interest.
analysis_slides <- meta[groupedPrimaryDiagnosisPatho %in% c("IM", "NIM", "NV"), as.character(slideId)]
analysis_keys <- unique(c(sample_map[as.character(slideId) %in% analysis_slides, sentrix_key], nevus_keys))
cat("Restricting CNV analysis to", length(analysis_keys), "samples (IM/NIM/NV)\n")

cat("=== Loading EPICv1 intensities ===\n")
mat_v1 <- load_intensities(path("idat_v1_dir"), analysis_keys)
cat("=== Loading EPICv2 intensities ===\n")
mat_v2 <- load_intensities(path("idat_v2_dir"), analysis_keys)

out_dir <- CNV_BURDEN_DIR
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

run_cnv_for_reference(
  mat_v1, mat_v2, nevus_keys, sample_map,
  genome_csv = file.path(out_dir, "cnv_burden_results.csv")
)
cat("Done.\n")
