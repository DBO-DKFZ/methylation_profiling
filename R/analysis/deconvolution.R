library(EpiSCORE)

source("lib/utils.R")

ref_broad  <- readRDS(path("episcore_ref_broad"))
ref_immune <- readRDS(path("episcore_ref_immune"))

run_deconvolution <- function(betas_mat, array_type) {
  cat("Computing gene-level betas for", array_type, "...\n")
  avbeta <- constAvBetaTSS(betas_mat, type = array_type)

  res_broad  <- wRPC(avbeta, ref_broad,  wth = 0.35, maxit = 200)$estF
  res_immune <- wRPC(avbeta, ref_immune, wth = 0.35, maxit = 200)$estF

  broad_non_immune <- setdiff(colnames(res_broad), "Immune")
  scaled_immune <- res_immune * res_broad[rownames(res_immune), "Immune"]
  cbind(res_broad[, broad_non_immune, drop = FALSE], scaled_immune)
}

estF_v1 <- run_deconvolution(load_betas(path("betas_v1")),                     "850k")
estF_v2 <- run_deconvolution(load_betas(path("betas_v2"), keep_suffix = TRUE), "EPICv2")

df_episcore <- rbind(
  data.table(slideId = rownames(estF_v1), as.data.table(estF_v1), array = "EPICv1"),
  data.table(slideId = rownames(estF_v2), as.data.table(estF_v2), array = "EPICv2")
)
df_episcore <- attach_diagnosis(df_episcore)

out_dir <- dirname(path("deconv_episcore"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fwrite(df_episcore, path("deconv_episcore"))
cat("Saved CSV to", path("deconv_episcore"), "\n")
cat("Done.\n")
