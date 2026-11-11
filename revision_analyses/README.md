# Revision analyses

Code and archived output for the analyses added to the manuscript during peer review. The
step-wise pipeline in `../scripts/` builds the models and runs the screen; the scripts here
interrogate those artifacts and were written to answer specific reviewer requests. They are
kept separate because they share their own helper module and read from a completed
`models_out/qsar_ml_YYYYMMDD_HHMMSS/` run rather than creating one.

Manuscript-preparation code used in the same revision (document editing, citation and style
audits, cover letter and highlights generation) is not part of this repository.

## Layout

- `scripts/` - the analyses, plus the shared helpers `R_common.py` and `R_ad.py`
- `figures/` - figure generation from the archived output in `results/`
- `docking/` - shell helpers for receptor preparation and the AutoDock Vina runs
- `results/analysis/` - per-analysis CSV and JSON output
- `results/tables/` - the condensed tables that feed the manuscript and supplement
- `results/docking_enrichment/` - docking scores, ligand manifest, prepared receptor, poses

## Configuration

`scripts/R_common.py` holds the paths every analysis uses: the project root, the model run
directory, and the output directories. The root defaults to the location used for the
published run and can be redirected without editing the file:

```bash
export NSD2_R1_ROOT=/path/to/your/checkout
```

The `MODELS` path points at the specific training run the manuscript reports
(`qsar_ml_20260412_162829`); change it there to analyze a different run. Every script adds
its own directory to `sys.path` and imports `R_common`, so each is run directly:

```bash
python scripts/R03_ad_weights_oof.py
```

## The three analyses raised at minor revision

- Applicability-domain weight grid: `scripts/R03_ad_weights_oof.py` recomputes the AD terms
  out-of-fold inside the development set for every scaffold partition and searches the
  (w1, w3) grid against development out-of-fold error, so the weights are chosen without
  touching the external sets. `scripts/R02_ad_weights_validate.py` checks the adopted
  configuration. Output: `results/analysis/R03_ad_weight_grid_all_seeds.csv`,
  `R03_ad_weight_grid_pooled.csv`, `R03_ad_weight_selection_per_seed.csv`, `R03_summary.json`.
- Property-matched putative-inactive set: `scripts/R11a_build_decoys.py` assembles it from the
  drug-like library inside the physicochemical windows, requiring topological dissimilarity to
  every active; `R11b_diagnose.py` reports matchability and coverage; `R11c_decoy_benchmark.py`
  scores each partition's model against the pool; `R11d_scores_for_curves.py` exports the
  per-compound scores behind the curves. Output: `results/analysis/R11_decoy_set.csv`,
  `R11_decoy_active_pairs.csv`, `R11_decoy_coverage.csv`, `R11_active_matchability.csv`,
  `R11_decoy_benchmark_per_seed.csv`, `R11_scores_for_curves.csv`, the `R11*_summary.json`
  files, and `results/tables/R11_decoy_benchmark_summary.csv`,
  `R11_decoy_property_match.csv`.
- Trajectory convergence diagnostics: `scripts/R09_md_convergence.py` computes, for every
  system, block averages of Ca RMSD over 25 ns windows, the integrated autocorrelation time
  and effective sample size, the split-half comparison of the MM/GBSA window, the PCA cosine
  content of the first two modes, and the drift of the running mean. Output:
  `results/analysis/R09_block_averages.csv`, `R09_summary.json`, and
  `results/tables/R09_md_convergence.csv`.

## The other analyses

- `R01_model_comparison_tests.py` - paired t-tests and Wilcoxon signed-rank tests between
  architectures over the shared partitions with Holm correction, the pooled exact McNemar
  test, and threshold stability and sensitivity.
- `R04_ad_threshold_error.py` - association between AD support and per-compound error,
  resolved by decile and swept across gate thresholds.
- `R05_size_descriptor_ablation.py`, `R05b_shap_size_free.py` - retraining with every
  size-related descriptor removed, and the resulting shift in SHAP attribution.
- `R06_scaffold_novelty.py` - Bemis-Murcko scaffold and generic framework counts for the
  prioritized set against the training scaffolds.
- `R07_tanimoto_space_check.py` - maximum Tanimoto similarity of the prioritized set to the
  training compounds in the filtered model space and in the full 2048-bit space.
- `R08_learning_curve.py` - scaffold-aware learning curve, whole scaffolds withheld at each
  training-set size.
- `R10_charges_and_mmgbsa_error.py` - ligand net formal charges alongside MM/GBSA component
  errors corrected for trajectory autocorrelation.
- `R12_rank_weight_sensitivity.py` - the multi-objective ranking weights scanned over the
  full simplex and over a plausible neighbourhood.
- `R13a_prepare_docking.py`, `R13b_run_docking.py`, `R13c_summarize_docking.py` - the
  retrospective docking benchmark: three-dimensional preparation of actives and matched
  decoys, the Vina runs against the rigid receptor, and the enrichment summary.
- `R14c_external_calibration.py` - the stored calibrators applied once to the untouched
  external sets.
- `R15_consistency_check.py` - reconciles the stereochemistry-independent and
  stereochemistry-aware scaffold conventions, and the two similarity spaces.

## Not archived here

- `results/docking_enrichment/ligands/` is omitted: 3,072 prepared PDBQT files, regenerated
  by `scripts/R13a_prepare_docking.py` from `ligand_manifest.csv` and the decoy set.
- Trajectory files from the molecular dynamics runs are too large to archive. The convergence
  diagnostics read the CPPTRAJ output produced by the input files in `../mds/`.
