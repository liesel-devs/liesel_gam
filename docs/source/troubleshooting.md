# Troubleshooting

## Problems with interfacing to R / Problems with ryp

We use the Python package [`ryp`](https://github.com/Wainberg/ryp) to interface to R.
It uses the R installation pointed to by the environment variable `R_HOME`, or if
`R_HOME` is not defined by running `R HOME`. If you encounter problems with the
interface to R, it may be a solution to set the `R_HOME` environment variable manually.

Here, as an example, we set `R_HOME` to an R installation in a conda environment:

```bash
export R_HOME=conda/envs/r-renv/lib/R
```

In some cases, you may also need to explicitly append the R library to the environment
variable `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
```
