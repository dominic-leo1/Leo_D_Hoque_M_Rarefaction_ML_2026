# Rarefaction is safe for classification but reshapes ASV biomarkers

Code and figures for a multi-disease benchmark of machine learning on 16S rRNA
microbiome data, testing how rarefaction preprocessing affects (i) disease
classification performance and (ii) SHAP-based biomarker discovery, across 12
publicly available datasets spanning 11 diseases and six body sites.

## Pipeline

```
 raw 16S data (SRA)
        │
        ▼
 [1] preprocessing (R)          scripts/01_preprocessing/preprocess_from_rds.R
        │   phyloseq objects ->  48 feature tables (TSV)
        ▼                        {nonrarefied,rarefied} x {asv,taxa}, 12 studies
 [2] classification (Python)    scripts/02_classification/run_pycaret_benchmark.py
        │   48 tables        ->  ml_summary_means.tsv (cross-validation metrics)
        ▼
 [3] SHAP feature importance    scripts/03_shap/
        │   tables + models  ->  shap_bodysite_*.csv (pooled per body site)
        ▼
 [4] figures & tables (Python)  scripts/04_figures/
            performance_figures.py   -> Fig 1-3, Supp Fig 1-7, 9
            biomarker_figures.py     -> Fig 4, 5, Supp Fig 8, 10
```

## Repository layout

```
scripts/
  01_preprocessing/preprocess_from_rds.R      label-free feature tables
  02_classification/run_pycaret_benchmark.py  PyCaret classification benchmark
  03_shap/                                    SHAP feature-importance stage (see its README)
  04_figures/
    performance_figures.py   classification figures and tables
    biomarker_figures.py     biomarker figures (entry point for Fig 4, 5, S8, S10)
    biomarker_core.py        biomarker figure/table functions
    robustness_analyses.py   top-K Jaccard sensitivity and SHAP robustness
data/                        derived inputs for the figure stage
docs/OUTPUTS.md              which script produces which figure and table
env/                         Python and R dependency lists
```

## Running

The figure stage runs directly from the derived inputs in `data/`:

```bash
make figures
# or, equivalently:
cd scripts/04_figures
python performance_figures.py     # Fig 1-3, Supp Fig 1-7, 9  -> ./output
python biomarker_figures.py       # Fig 4, 5, Supp Fig 8, 10  -> ./output
```

The preprocessing and classification stages operate on the raw sequence data
and the intermediate feature tables. Set the input and output directories at
the top of each script, then:

```bash
Rscript scripts/01_preprocessing/preprocess_from_rds.R
python  scripts/02_classification/run_pycaret_benchmark.py
```

## Environment

- Python 3.11+. Install `env/requirements-figures.txt` for the figure stage and
  `env/requirements-classification.txt` for the benchmark. The benchmark uses
  PyCaret 3.3.2 with joblib pinned to 1.4.2.
- R 4.6+. See `env/R-packages.txt`.

## Data

Raw sequences are available from the NCBI SRA (accessions in the manuscript's
`Table 1` and `Table S1`). The large intermediate feature tables and trained
models are not stored here; the compact derived inputs needed to reproduce the
figures (`ml_summary_means.tsv`, `shap_bodysite_*.csv`) are provided in `data/`.

## Citation

See `CITATION.cff`.

## License

See `LICENSE`.
