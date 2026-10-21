# ConFlow-JEPA CFD runner

Public companion repository for SU2 runs of the ConFlow-JEPA project.
It holds only the SU2 configuration writer, run workflow (GitHub Actions) and per-case input files.
Model code, data and the manuscript live in a separate private repository.

SU2 is installed from conda-forge at run time. Each case runs as one Actions job; runs longer than
the 6 h job limit continue from the saved solution in a chained job.
