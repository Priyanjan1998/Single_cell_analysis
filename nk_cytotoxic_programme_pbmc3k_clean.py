import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

sc.settings.verbosity = 1
sc.set_figure_params(dpi=100, facecolor="white", frameon=False)
SEED = 0
np.random.seed(SEED)

# ---- 1. Load data
adata = sc.datasets.pbmc3k()   # 10x Genomics PBMC, raw counts
adata.var_names_make_unique()
print(adata)

# ---- 2. Quality control
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
             jitter=0.4, multi_panel=True)

n0 = adata.n_obs
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[(adata.obs["n_genes_by_counts"] < 2500) & (adata.obs["pct_counts_mt"] < 5)].copy()
print(f"cells: {n0} -> {adata.n_obs}; genes: {adata.n_vars}")

# ---- 3. Normalise, select variable genes, reduce dimensions, cluster
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata                      # keep all genes (log-normalised) for plotting and testing

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver="arpack", random_state=SEED)
sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=30, random_state=SEED)
sc.tl.umap(adata_hvg, random_state=SEED)
sc.tl.leiden(adata_hvg, resolution=0.5, flavor="igraph", n_iterations=2,
             directed=False, random_state=SEED)

adata.obs["leiden"] = adata_hvg.obs["leiden"]
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
sc.pl.umap(adata, color="leiden", legend_loc="on data")

# ---- 4. Annotate clusters from canonical markers
lineage_markers = {
    "B":           ["MS4A1", "CD79A", "CD79B"],
    "CD14 Mono":   ["CD14", "LYZ", "S100A9"],
    "FCGR3A Mono": ["LST1", "AIF1", "COTL1"],
    "DC":          ["FCER1A", "CST3"],
    "Platelet":    ["PPBP", "PF4"],
    "CD4 T":       ["IL7R", "CD3D", "CD3E", "LTB"],
    "CD8 T":       ["CD8A", "CD8B", "CD3D", "CD3E"],
    "NK":          ["GNLY", "KLRD1", "KLRF1", "NCAM1", "FCGR3A"],
}
present = {k: [g for g in v if g in adata.raw.var_names] for k, v in lineage_markers.items()}
missing = {k: sorted(set(v) - set(present[k])) for k, v in lineage_markers.items() if set(v) - set(present[k])}
print("markers not found in this dataset:", missing)

expr = adata.raw.to_adata().to_df()
cluster_mean = expr.groupby(adata.obs["leiden"], observed=True).mean()
scores = pd.DataFrame({k: cluster_mean[v].mean(axis=1) for k, v in present.items() if v})
z = (scores - scores.mean()) / scores.std(ddof=0)
auto = z.idxmax(axis=1)
print(auto.to_string())

overrides = {}          # e.g. {"5": "NK"} after checking the dot plot below
labels = {**auto.to_dict(), **overrides}
adata.obs["cell_type"] = adata.obs["leiden"].map(labels).astype("category")

sc.pl.dotplot(adata, {k: v for k, v in present.items() if v}, groupby="leiden",
              standard_scale="var", dendrogram=False)
sc.pl.umap(adata, color="cell_type")
print(adata.obs["cell_type"].value_counts())

check_genes = [g for g in ["CD3D", "CD3E", "CD8A", "GNLY"] if g in adata.raw.var_names]
check = expr[check_genes].groupby(adata.obs["cell_type"], observed=True).mean().round(2)
print(check)
if "NK" in check.index and "CD3D" in check.columns and check.loc["NK", "CD3D"] > check["CD3D"].median():
    print("WARNING: the 'NK' label has high CD3D. Re-check the clusters and use `overrides`.")

# ---- 5. Cytotoxic machinery across cell types
panel = {
    "Effectors":           ["PRF1", "GZMA", "GZMB", "GZMH", "GZMK", "GNLY", "NKG7"],
    "Granule trafficking": ["UNC13D", "STX11", "STXBP2", "LAMP1"],
    "Protection":          ["SERPINB9", "CTSB"],
    "Regulators":          ["RUNX3", "EOMES", "TBX21", "IFNG"],
}
panel = {k: [g for g in v if g in adata.raw.var_names] for k, v in panel.items()}
print({k: v for k, v in panel.items()})

sc.pl.dotplot(adata, panel, groupby="cell_type", standard_scale="var")

genes = [g for v in panel.values() for g in v]
frac = (adata.raw.to_adata()[:, genes].X > 0)
frac = pd.DataFrame(np.asarray(frac.todense()) if hasattr(frac, "todense") else frac,
                    columns=genes, index=adata.obs_names)
pct = (frac.groupby(adata.obs["cell_type"], observed=True).mean() * 100).round(1)
print(pct.T)

# ---- 6. NK vs CD8 T differential expression and volcano plot
assert {"NK", "CD8 T"} <= set(adata.obs["cell_type"]), "NK or CD8 T not assigned: check section 4"
sub = adata[adata.obs["cell_type"].isin(["NK", "CD8 T"])].copy()
sc.tl.rank_genes_groups(sub, groupby="cell_type", groups=["NK"], reference="CD8 T",
                        method="wilcoxon", corr_method="benjamini-hochberg")
de = sc.get.rank_genes_groups_df(sub, group="NK")
de["minus_log10_padj"] = -np.log10(de["pvals_adj"].clip(lower=1e-300))
print(f"NK cells: {(sub.obs.cell_type == 'NK').sum()}, CD8 T cells: {(sub.obs.cell_type == 'CD8 T').sum()}")
print(de.head(10))

FC, P = 1.0, 0.05
de["class"] = np.select(
    [(de.logfoldchanges >= FC) & (de.pvals_adj < P), (de.logfoldchanges <= -FC) & (de.pvals_adj < P)],
    ["Higher in NK", "Higher in CD8 T"], default="NS")
colours = {"Higher in NK": "#d62728", "Higher in CD8 T": "#1f77b4", "NS": "#c8c8c8"}

fig, ax = plt.subplots(figsize=(7, 6))
for k, c in colours.items():
    d = de[de["class"] == k]
    ax.scatter(d.logfoldchanges, d.minus_log10_padj, s=10, c=c, label=f"{k} (n={len(d)})", alpha=.8, linewidths=0)
ax.axvline(FC, ls="--", c="k", lw=.8); ax.axvline(-FC, ls="--", c="k", lw=.8)
ax.axhline(-np.log10(P), ls="--", c="k", lw=.8)

highlight = [g for g in genes if g in set(de.names)]
for _, r in de[de.names.isin(highlight)].iterrows():
    ax.annotate(r["names"], (r.logfoldchanges, r.minus_log10_padj), fontsize=8,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("log2 fold change (NK vs CD8 T)")
ax.set_ylabel("-log10 adjusted p-value")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig("nk_vs_cd8t_volcano.png", dpi=150)
plt.show()

cols = ["names", "logfoldchanges", "pvals_adj"]
print("Top genes higher in NK:");      print(de[de["class"] == "Higher in NK"].sort_values("logfoldchanges", ascending=False)[cols].head(10).to_string(index=False))
print("\nTop genes higher in CD8 T:"); print(de[de["class"] == "Higher in CD8 T"].sort_values("logfoldchanges")[cols].head(10).to_string(index=False))
print("\nCytotoxic panel genes:");     print(de[de.names.isin(genes)][cols].round(4).to_string(index=False))
