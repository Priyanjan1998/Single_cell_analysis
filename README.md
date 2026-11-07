# Single-cell analysis of the cytotoxic gene programme (public PBMC data)

`nk_cytotoxic_programme_pbmc3k.ipynb` analyses the public 10x Genomics PBMC 3k dataset (about 2,700 cells, one donor) with Scanpy: quality control, normalisation, highly variable genes, PCA, UMAP, Leiden clustering, marker-based annotation, a cytotoxic gene panel (perforin, granzymes, FHL granule-trafficking genes, perforin regulators) and NK vs CD8 T differential expression with a volcano plot.

Run it:

```bash
pip install -r requirements.txt
jupyter notebook nk_cytotoxic_programme_pbmc3k.ipynb
```

The first run downloads the dataset automatically.

Limitations: one donor, peripheral blood (not iPSC-derived NK cells), marker-based annotation that should be checked against the dot plot, and tutorial-based QC thresholds. The findings section at the end of the notebook is left for you to write after running it.
