# Absence spares, substitution diverts: reproducibility code

Code and supplied analysis inputs for the manuscript **“Absence spares, substitution diverts: a two-timescale plasticity model of expert skill under disuse and AI assistance.”**

## Contents

- `model.py` — core two-pathway model with two-timescale consolidation and gain arbitration.
- `fit_absence.py` — fits the absence branch to the supplied estimates and absence-gap summary data.
- `sim_substitution.py` — simulations for learners, experts, recovery, and reserve; writes simulation results and cached expert states.
- `figures.py` — generates figures from fit and simulation outputs.
- `00_fide_bin_gaps.py` — derives mean absence-gap lengths from the FIDE monthly panel.
- `fill_numbers.py` — inserts computed values into the supplied manuscript template.
- `fide_results.json` — supplied summary estimates from the companion analysis.
- `ms_template.md` — manuscript text template used by `fill_numbers.py`.

## Important data limitation

The FIDE monthly panel file required by `00_fide_bin_gaps.py` was **not included** among the files supplied for this package. Consequently, the first-stage FIDE analysis cannot be rerun from this package alone. The file `fide_results.json` is included, but it does not replace the panel needed to calculate `fide_bin_mean_gap.csv`.

Before public release, authors should either (a) add the permitted, appropriately documented FIDE panel/input file, or (b) provide a stable source and precise instructions for obtaining it, subject to the source's terms and applicable data restrictions. Do not describe the package as fully end-to-end reproducible until this dependency is resolved and the workflow has been tested from a clean environment.

## Requirements

Python 3.10 or later is recommended. Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The FIDE-panel step additionally requires a Parquet reader such as `pyarrow` and the panel file described above.

## Workflow

Run commands from this repository directory. The first step needs the external panel file and an output directory:

```bash
mkdir -p out
python 00_fide_bin_gaps.py /path/to/fide_panel.parquet out
```

This produces `out/fide_bin_mean_gap.csv`. Then run:

```bash
python fit_absence.py fide_results.json out/fide_bin_mean_gap.csv out
python sim_substitution.py out
python figures.py out
python fill_numbers.py ms_template.md out/fit_absence.json out/sim_results.json fide_results.json ms.md
```

The simulation step can take several minutes. To run the script's reduced/fast configuration, use:

```bash
python sim_substitution.py out fast
```

The fitting, figure-generation, and manuscript-number insertion steps expect outputs from preceding steps. Run the full workflow in order.

## Reproducibility notes

- Simulation seeds are specified in the scripts; see `sim_substitution.py`.
- Generated files are written to the chosen output directory and are not included here.
- Test the workflow in a fresh Python environment before submitting the repository URL to the journal.
- Record the exact Python and dependency versions used for the final reproduction run.

## Citation and licence

If you use this code, cite the associated manuscript. The authors should select and add an explicit software licence before public release. No licence is asserted by this package as currently prepared.
