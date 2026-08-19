# Apptainer images

Two single-purpose images for running this project:

| Image        | Definition   | Contents                                                                                               |
|--------------|--------------|--------------------------------------------------------------------------------------------------------|
| `r.sif`      | `r.def`      | R 4.6 / Bioconductor 3.23: `sesame`, `methyLImp2`, `DMRcate`, `conumee2`, `EpiSCORE`, `methylclock`, … |
| `python.sif` | `python.def` | Python 3.14 (from `python/uv.lock`): `torch`, `xgboost`, `scikit-learn`, `statsmodels`, …               |

## Build (locally)

```bash
./apptainer/build.sh          # both images
./apptainer/build.sh python   # just one
```

> `python.sif` is multi-GB because `torch` resolves to the CUDA build on Linux.


## sesame annotation cache

The imputation step (`methyLImp2`) needs no downloads. 
Steps using `sesame`/`openSesame` fetch annotation data on first use into `$HOME/.cache`. 
Pre-populate once with network access:

```bash
apptainer exec "$ROOT/r.sif" Rscript -e 'sesameData::sesameDataCacheAll()'
```
