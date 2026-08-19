library(sesame)
library(BiocParallel)

source("lib/utils.R")

# EPICv1
idat_dir_v1 <- path("epic_v1_dir")

qcs <- openSesame(idat_dir_v1,
                 prep = '',
                 BPPARAM = MulticoreParam(workers = config$workers),
                 func = sesameQC_calcStats,
                 funs = "detection")

vec_v1 <- vapply(qcs, function(x) x@stat$frac_dt, numeric(1))

# EPICv2
idat_dir_v2 <- path("epic_v2_dir")

qcs_v2 <- openSesame(idat_dir_v2,
                 prep = '',
                 BPPARAM = MulticoreParam(workers = config$workers),
                 func = sesameQC_calcStats,
                 funs = "detection")

vec_v2 <- vapply(qcs_v2, function(x) x@stat$frac_dt, numeric(1))

fwrite(data.table(slideId = names(vec_v1), detection_rate = as.numeric(vec_v1)), path("epicv1_detectionrate"))
fwrite(data.table(slideId = names(vec_v2), detection_rate = as.numeric(vec_v2)), path("epicv2_detectionrate"))
cat("Done.\n")
