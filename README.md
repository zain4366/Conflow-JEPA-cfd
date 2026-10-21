# ConFlow-JEPA CFD runner

Public companion repository for the SU2 runs of the ConFlow-JEPA project. It holds only the SU2
runner (config writer, meshing, parsers, convergence rule), the GitHub Actions workflow and the
case definitions. Model code, training data and the manuscript live in a separate private repository;
`runner/conflow/` is a vendored subset of it (source commit and SHA-256 in `runner/VENDORED.json`).

## Running cases
* Edit `requests/run.json` on `main` (list of case folders, `"reference": true` to also download the
  experimental data), or use *Actions > su2 > Run workflow*.
* Each case runs in its own job: SU2 8.5.0 and gmsh are installed from conda-forge with micromamba
  (`env/su2-env.yml`). SU2 runs in segments restarted from its own solution; the PLAN convergence
  rule (log10 density residual drop >= 5, or CL change < 1e-4 over 200 iterations) is checked between
  segments. A case that reaches the 330 min job budget continues in the next chained job (up to 4).
* Results are committed to the branch `results`: `latest/<case>/` and `runs/<run_id>/<case>/` with
  `status.json` (status, convergence, timing, machine, SU2 version, mesh provenance), `history.csv`,
  `surface_flow.csv`, the last config and log excerpts. Failed and unconverged runs are kept.

## Cases
| case | what |
|---|---|
| m6_euler | ONERA M6, M 0.8395, AoA 3.06 deg, Euler, SU2 TestCases mesh |
| m6_rans_hexa | same, RANS-SA, 43,008-cell hexahedral TestCases mesh |
| m6_rans_hybrid | same, RANS-SA, 258,969-cell hybrid TestCases mesh (cost basis) |
| rae2822_c6 | RAE 2822 case 6 (M 0.729, AoA 2.31 deg, Re 6.5e6), RANS-SA, TestCases mesh |
| rae2822_c6_ogrid | same on the ConFlow 2D O-grid |
| sw_holdout_euler | one held-out SuperWing test wing, Euler, gmsh mesh |

## Data and licences
* Meshes: github.com/su2code/TestCases (pinned commit in each `case.json`).
* Experimental data: NASA NPARC Alliance validation archive (ONERA M6, test 2308; RAE 2822 case 6,
  AGARD AR-138), downloaded at run time with URL and SHA-256 in `reference/manifest.json`.
* SuperWing: Yang et al. (2025), huggingface.co/datasets/yunplus/SuperWing, CC BY-SA 4.0. The
  surface and Cp excerpts of the held-out wing in `results` are under the same licence.
* SU2 (LGPL-2.1, conda-forge build GPL-2.0-or-later); gmsh (GPL-2.0-or-later).
