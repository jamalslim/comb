# Provenance of the release files

Each file of the release, the file it was made from (its path in the build tree, and the working tree that file came
from), what was changed, and the first 16 hex digits of the SHA-256 of the source. "import paths" means only the
module paths of `from comb...` statements were changed to `comb.showers` or `comb.digits`; "paths for scripts/"
means the file's own location or the location of a driver it calls, now in `scripts/`. "one statement per line" and
"docstrings" change the layout and the documentation only: each file's syntax tree, docstrings aside, is identical to
the source's after the other listed changes.

| File | Made from | Changes | SHA-256 of the source |
| --- | --- | --- | --- |
| `ALGORITHM.md` | `ALGORITHM.md` in comb3 (uploaded) | documentation or commands | `3910d768aa63a217` |
| `comb/digits/__init__.py` | `comb2/comb/__init__.py` in comb_v99 (digits) | moved, documentation or commands | `dea6701ef9d08e77` |
| `comb/digits/circuit.py` | `comb2/comb/circuit.py` in comb_v99 (digits) | moved, one statement per line | `043679a14eebe663` |
| `comb/digits/hqmm.py` | `comb2/comb/hqmm.py` in comb_v99 (digits) | moved, one statement per line | `b5b35226ca9cb27f` |
| `comb/digits/kpoint.py` | `comb2/comb/kpoint.py` in comb_v99 (digits) | moved, one statement per line | `993f4ebfca46e11a` |
| `comb/digits/mmd.py` | `comb2/comb/mmd.py` in comb_v99 (digits) | moved | `0a7ed26fcfd2f9f6` |
| `comb/digits/model.py` | `comb2/comb/model.py` in comb_v99 (digits) | moved, one statement per line | `3d35969528001145` |
| `comb/digits/noise.py` | `comb2/comb/noise.py` in comb_v99 (digits) | moved, one statement per line | `02ae2c0be947bd05` |
| `comb/digits/noisytree.py` | `comb2/comb/noisytree.py` in comb_v99 (digits) | moved, one statement per line | `bde2019f9d81ea2e` |
| `comb/digits/pencf.py` | `comb2/comb/pencf.py` in comb_v99 (digits) | moved, one statement per line | `589280502ed2cc1e` |
| `comb/digits/strokes.py` | `comb2/comb/strokes.py` in comb_v99 (digits) | moved, one statement per line | `cccf51313994b74a` |
| `comb/digits/train.py` | `comb2/comb/train.py` in comb_v99 (digits) | moved, one statement per line | `b8bc68ccf6ae8df1` |
| `comb/digits/treecomb.py` | `comb2/comb/treecomb.py` in comb_v99 (digits) | moved, one statement per line | `b3d57d580d5c9ebd` |
| `comb/showers/__init__.py` | `comb/__init__.py` in Maxwell tree (newest shower code) | moved, documentation or commands | `ac8d6eb801cde7e3` |
| `comb/showers/circuit.py` | `comb/circuit.py` in Maxwell tree (newest shower code) | moved, one statement per line | `1ad2b635072feb91` |
| `comb/showers/data.py` | `comb/data.py` in Maxwell tree (newest shower code) | moved, one statement per line | `e9b4173cd1f9360d` |
| `comb/showers/envmem.py` | `comb/envmem.py` in Maxwell tree (newest shower code) | moved, one statement per line | `457240d6e7f5eaf3` |
| `comb/showers/evaluate.py` | `comb/evaluate.py` in Maxwell tree (newest shower code) | moved | `4061658532a28794` |
| `comb/showers/mmd.py` | `comb/mmd.py` in Maxwell tree (newest shower code) | moved | `0a7ed26fcfd2f9f6` |
| `comb/showers/model.py` | `comb/model.py` in Maxwell tree (newest shower code) | moved, one statement per line | `f08e07b0cbcfeb84` |
| `comb/showers/noise.py` | `comb/noise.py` in Maxwell tree (newest shower code) | moved, one statement per line | `7d80553f0c4f24b0` |
| `comb/showers/plots_ds1.py` | `comb/plots_ds1.py` in Maxwell tree (newest shower code) | moved, one statement per line | `6512c8647259934a` |
| `comb/showers/qiskit_comb.py` | `comb/qiskit_comb.py` in Maxwell tree (newest shower code) | moved, one statement per line | `8fa30a56764aebb1` |
| `comb/showers/train.py` | `comb/train.py` in Maxwell tree (newest shower code) | moved, one statement per line | `b8bc68ccf6ae8df1` |
| `comb/showers/zoo.py` | `comb/zoo.py` in Maxwell tree (newest shower code) | moved | `e1ceebea879810e1` |
| `scripts/comb_ds1.py` | `comb_ds1.py` in Maxwell tree (newest shower code) | moved, import paths, paths for scripts/, location checks for comb/showers/, one statement per line, docstrings | `b4f4f0fdb0a7373a` |
| `scripts/comb_strokes.py` | `comb2/comb_strokes.py` in comb_v99 (digits) | moved, import paths, paths for scripts/, one statement per line, docstrings | `97a23e85cf8d2327` |
| `docs/digits_experiments.md` | `comb2/RUN_PAPER2.md` in comb_v99 (digits) | moved, documentation or commands | `09d32521cbb35816` |
| `docs/digits_method.md` | `comb2/STROKES_METHOD.md` in comb_v99 (digits) | moved, documentation or commands | `50b5bfccd5af2ca7` |
| `docs/digits_overview.md` | `comb2/README.md` in comb_v99 (digits) | moved, documentation or commands | `d6b7c004ce33413f` |
| `docs/double_peak_fix.md` | `docs/double_peak_fix.md` in comb_v99 README_axis_fix.md | documentation or commands | `11da6dbbcfdfe635` |
| `docs/hardware_compatible_model.md` | `docs/hardware_compatible_model.md` in comb_v99 README_counts_compatible.md | documentation or commands | `e270f6f5c058800c` |
| `scripts/make_paper_numbers.py` | `comb2/make_paper_numbers.py` in comb_v99 (digits) | moved, paths for scripts/, one statement per line | `0d0eef04281afc6a` |
| `maxwell/README.md` | `maxwell/README.md` in Maxwell tree | documentation or commands | `c2485656e68a92f9` |
| `maxwell/bootstrap_equiv.py` | `maxwell/bootstrap_equiv.py` in Maxwell tree | one statement per line | `36f5897e247c012c` |
| `maxwell/converge.py` | `maxwell/converge.py` in Maxwell tree | paths for scripts/, one statement per line | `b71819c8956ddafc` |
| `maxwell/converge.sbatch` | `maxwell/converge.sbatch` in Maxwell tree | documentation or commands | `24c6f2a83cab9734` |
| `maxwell/env.sbatch` | `maxwell/env.sbatch` in Maxwell tree | documentation or commands | `84ca343cc0259458` |
| `maxwell/env.sh` | `maxwell/env.sh` in Maxwell tree | unchanged | `13049914d20db550` |
| `maxwell/k16.sbatch` | `maxwell/k16.sbatch` in Maxwell tree | documentation or commands | `e6b9c64054f1020d` |
| `maxwell/make_runs.py` | `maxwell/make_runs.py` in Maxwell tree | one statement per line | `ecbc5551134fbd03` |
| `maxwell/memory_usage.py` | `maxwell/memory_usage.py` in Maxwell tree | import paths, one statement per line | `ea91a57b0aebd2b6` |
| `maxwell/pilot_lr.sbatch` | `maxwell/pilot_lr.sbatch` in Maxwell tree | documentation or commands | `bc9a7b7773903e13` |
| `maxwell/plot_memory_curve.py` | `maxwell/plot_memory_curve.py` in Maxwell tree | one statement per line | `1c817d1f94ff5f41` |
| `maxwell/plot_results.py` | `maxwell/plot_results.py` in Maxwell tree | one statement per line | `f322f97a71075a8d` |
| `maxwell/runs_hmm_curve.txt` | `maxwell/runs_hmm_curve.txt` in Maxwell tree | unchanged | `3d419797193c165a` |
| `maxwell/runs_main.txt` | `maxwell/runs_main.txt` in Maxwell tree | unchanged | `3853ce49405eab63` |
| `maxwell/seeds.sbatch` | `maxwell/seeds.sbatch` in Maxwell tree | documentation or commands | `65d308530cd9447c` |
| `maxwell/summarize_seeds.py` | `maxwell/summarize_seeds.py` in Maxwell tree | import paths, paths for scripts/, one statement per line, docstrings | `dc0b2c06dda7d420` |
| `maxwell/summary.sbatch` | `maxwell/summary.sbatch` in Maxwell tree | documentation or commands | `9fd076d186e1830c` |
| `maxwell/warm.sbatch` | `maxwell/warm.sbatch` in Maxwell tree | documentation or commands | `6a3bf8f15400c9c5` |
| `maxwell/warm6000.sbatch` | `maxwell/warm6000.sbatch` in Maxwell tree | documentation or commands | `6ffb8eb4276f3c19` |
| `plot/README.md` | `plot/README.md` in Maxwell tree = comb_v99 (identical) | documentation or commands | `b909dcc48238c1b1` |
| `plot/digits_plots.py` | `plot/digits_plots.py` in Maxwell tree = comb_v99 (identical) | import paths, paths for scripts/, one statement per line | `ad39daa72896412e` |
| `plot/ds1_plots.py` | `plot/ds1_plots.py` in Maxwell tree = comb_v99 (identical) | one statement per line | `9f8737b077e76207` |
| `plot/hepstyle.py` | `plot/hepstyle.py` in Maxwell tree = comb_v99 (identical) | one statement per line | `19540927a6403893` |
| `plot/make_all.py` | `plot/make_all.py` in Maxwell tree = comb_v99 (identical) | one statement per line | `5023c98d54644a2e` |
| `scripts/class_quality.py` | `comb2/scripts/class_quality.py` in comb_v99 (digits) | moved, import paths, one statement per line | `c2addd255bffe025` |
| `scripts/deploy_strokes.py` | `comb2/scripts/deploy_strokes.py` in comb_v99 (digits) | moved, import paths, one statement per line | `e724dfea8b467ccf` |
| `scripts/finetune_readout.py` | `scripts/finetune_readout.py` in comb_v99 (hardware-compatible shower model) | import paths, one statement per line | `59124d4d37b490d6` |
| `scripts/hardware_run.py` | `scripts/hardware_run.py` in Maxwell tree | import paths, paths for scripts/, one statement per line, docstrings | `4e624069b100c99d` |
| `scripts/memory_equiv.py` | `comb2/scripts/memory_equiv.py` in comb_v99 (digits) | moved, one statement per line | `6fbae09bb5fd460a` |
| `scripts/paper2.py` | `comb2/scripts/paper2.py` in comb_v99 (digits) | moved, paths for scripts/, one statement per line | `448b0050553a3375` |
| `scripts/quantile_readout.py` | `scripts/quantile_readout.py` in comb_v99 (hardware-compatible shower model) | import paths, one statement per line | `8c9eec4a98576a0b` |
| `scripts/recalibrate_readout.py` | `scripts/recalibrate_readout.py` in comb_v99 (hardware-compatible shower model) | import paths, paths for scripts/, location checks for comb/showers/, one statement per line | `d0167c7ad71541ea` |
| `scripts/statistical_errors.py` | `comb2/scripts/statistical_errors.py` in comb_v99 (digits) | moved, import paths, one statement per line | `6c5241744e979395` |
| `scripts/strokes_table.py` | `comb2/scripts/strokes_table.py` in comb_v99 (digits) | moved, one statement per line | `76347ee9652b09b8` |
| `scripts/test_comb.py` | `scripts/test_comb.py` in Maxwell tree | import paths, one statement per line | `e0abbaea733128f8` |
| `scripts/test_hqmm.py` | `comb2/scripts/test_hqmm.py` in comb_v99 (digits) | moved, import paths, one statement per line | `5a7b6c130ae44e25` |
| `scripts/test_kpoint.py` | `comb2/scripts/test_kpoint.py` in comb_v99 (digits) | moved, import paths, one statement per line | `8d030e0be19001d1` |
| `scripts/test_kraus.py` | `scripts/test_kraus.py` in Maxwell tree | import paths, one statement per line | `dc1b0cd619a706c5` |
| `scripts/test_noise.py` | `scripts/test_noise.py` in Maxwell tree | import paths, one statement per line | `1df4e1fa087b167a` |
| `scripts/test_noisytree.py` | `comb2/scripts/test_noisytree.py` in comb_v99 (digits) | moved, import paths, one statement per line | `758e3e09f0ddc44e` |
| `scripts/test_pencf.py` | `comb2/scripts/test_pencf.py` in comb_v99 (digits) | moved, import paths, one statement per line | `a65ac4a7c6ebfb54` |
| `scripts/test_strokes.py` | `comb2/scripts/test_strokes.py` in comb_v99 (digits) | moved, import paths, one statement per line | `6dfd9c9e971b49f9` |
| `scripts/test_train.py` | `scripts/test_train.py` in Maxwell tree | import paths, one statement per line | `db1630f4d0bc9005` |
| `scripts/test_treecomb.py` | `comb2/scripts/test_treecomb.py` in comb_v99 (digits) | moved, import paths, one statement per line | `ed092fd72de1c429` |
