# mgcv example-data assets

`columb.csv` and `columb_polys.csv` are committed copies of the Columbus data shipped
with mgcv. Normal tests only read these files and require neither R nor mgcv.

Regenerate the assets with an mgcv installation using:

```console
Rscript tests/mgcv_data/generate.R tests/mgcv_data
```
