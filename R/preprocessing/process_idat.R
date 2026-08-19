library(sesame)
library(BiocParallel)
library(DMRcate)

source("lib/utils.R")

# One beta-value CSV per IDAT batch directory.
process_idat_directories <- function(idat_dir, output_dir, suffix = '_betas.csv', workers = config$workers, prep = 'QCDPB') {
  for (d in list.dirs(idat_dir, full.names = TRUE, recursive = FALSE)) {
    betas <- openSesame(d,
                        prep = prep,
                        BPPARAM = MulticoreParam(workers = workers),
                        func = getBetas)

    # Drop SNP, cross-hybridising and sex-chromosome probes, then the control and rs probes.
    betas <- rmSNPandCH(betas,
                        rmcrosshyb = TRUE,
                        rmXY = TRUE)
    betas <- betas[!grepl('^(ctl|rs)', rownames(betas)),]

    output_file <- file.path(output_dir, paste0(basename(d), suffix))
    fwrite(as.data.table(betas, keep.rownames = TRUE), output_file)
  }
}

process_idat_directories(
  idat_dir = path("idat_v1_dir"),
  output_dir = path("epic_v1_dir"),
  suffix = '_betas.csv'
)

process_idat_directories(
  idat_dir = path("idat_v2_dir"),
  output_dir = path("epic_v2_dir"),
  suffix = '_betas_v2.csv'
)
