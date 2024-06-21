# Non-negative Universal Differential Equations (nUDE)

This repository contains the implementation of non-negative and regularised universal differential equations (UDEs) and supplemental material accompanying the 2024 FOSBE publication "Non-Negative Universal Differential Equations With Applications in Systems Biology" [1]. It builds on the PEtab package for specification of a parameter estimation problem, AMICI for simulation and pyPESTO as an interface for parameter estimation.

The Python environment package versions that we used are stored in `environment/requirements.txt`, and can be installed with `pip install -r environment/requirements.txt`. A Python virtual environment is recommended, in case you want to use a fresh environment with the latest package versions later.

Computations were performed on an HPC, with the system packages described in `environment/env.sh`. These system dependencies are only required by the Python environment packages -- if you can install AMICI [2], you probably don't need to install additional system packages.

We additionally used the package provided in `nnUDE`. From inside the `nnUDE` directory where the `setup.py` file is, this package can be installed with `pip install -e .`.

The example models from the paper are provided in `examples`. Some are derived from the PEtab Benchmark Collection [3].

[1] arxiv preprint https://arxiv.org/abs/2406.14246  
[2] https://amici.readthedocs.io/en/latest/python_installation.html  
[3] https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab  

