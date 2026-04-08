ECSE343 — Group 32: nonlinear circuit simulation, parameter estimation, and ML

What’s here
  group_32_circuit_simulator.py — MNA + Backward Euler, Newton–Raphson per step, sensitivities, Gauss–Newton parameter fit.
  group_32_helper_functions.py — dataset generation (random R, C in project ranges), plotting, save/load pickle.
  test.py — end-to-end demo: forward simulation, Gauss–Newton on measurements.csv, optional dataset build to data/dataset.pkl.
  group_32_ML.ipynb — supervised models (linear/ridge, RF, SVR, MLP) vs Gauss–Newton comparison on the pickle dataset.
  data/measurements.csv — T×4 matrix, columns [V1, V2, V3, IE] (one row per time sample).

Setup
  python3 -m venv .venv && source .venv/bin/activate   # recommended on macOS/Homebrew Python
  pip install -r requirements.txt

Run
  python test.py

Ensure data/measurements.csv exists (and data/ for dataset.pkl output) before running.