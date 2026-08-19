# Annotates the EPIC manifests with Roadmap/ENCODE ChromHMM states, producing the CpG -> state lookup that the
# `chrom_hmm_*` filters and the ChromHMM reducer in python/ml/ consume. Reference E059 = primary foreskin melanocytes,
# the closest available match to the melanocytic lesions profiled here.
#
# Inputs : paths.manifest_v1 / paths.manifest_v2  (probe IDs with lifted hg38 coordinates)
#          paths.chrom_hmm_segments               (E059 15-state coreMarks segmentation, hg38-lifted BED)
# Outputs: paths.chrom_hmm (EPICv1), paths.chrom_hmm_v2 (EPICv2)
#          [seqnames, start, end, width, strand, Probe_ID, ChromHMM_E059_15]

library(rtracklayer)
library(GenomicRanges)

source("lib/utils.R")

chromhmm <- import(path("chrom_hmm_segments"))
cat("Loaded", length(chromhmm), "ChromHMM segments from", path("chrom_hmm_segments"), "\n")

annotate_manifest <- function(manifest_path, output_path, label) {
  cat("\n=== ChromHMM annotation for", label, "===\n")
  manifest <- fread(manifest_path, select = c("CpG_chrm", "CpG_beg", "CpG_end", "Probe_ID"))

  # Probes without lifted hg38 coordinates cannot be placed on the segmentation.
  n_total <- nrow(manifest)
  manifest <- manifest[!is.na(CpG_chrm) & !is.na(CpG_beg) & !is.na(CpG_end)]
  cat("Mappable probes:", nrow(manifest), "of", n_total, "\n")

  probes_gr <- GRanges(
    seqnames = manifest$CpG_chrm,
    ranges   = IRanges(start = manifest$CpG_beg, end = manifest$CpG_end),
    Probe_ID = manifest$Probe_ID
  )

  # A probe falling outside every segment keeps NA. Segments are non-overlapping, so at most one hit per probe.
  hits <- findOverlaps(probes_gr, chromhmm)
  states <- rep(NA_character_, length(probes_gr))
  states[queryHits(hits)] <- mcols(chromhmm)$name[subjectHits(hits)]
  mcols(probes_gr)$ChromHMM_E059_15 <- states
  cat("Annotated", sum(!is.na(states)), "probes with a ChromHMM state\n")

  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  fwrite(as.data.frame(probes_gr), output_path)
  cat("Wrote", output_path, "\n")
}

annotate_manifest(path("manifest_v1"), path("chrom_hmm"),    "EPICv1")
annotate_manifest(path("manifest_v2"), path("chrom_hmm_v2"), "EPICv2")
cat("\nDone.\n")
