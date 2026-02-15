# 🛠️ ECSE 343 – Group Project Roadmap

## 🗓️ Week 1 — Circuit Simulator Foundations
**Goal:** Time-domain simulation works and is stable

- [x] Read project statement fully (once, no overthinking)
- [x] Implement `getG(R)`
- [x] Implement `getC(C)`
- [x] Implement diode current `ID(V2, V3)`
- [x] Derive Jacobian `df/dx`
- [x] Implement `get_jac()` function
- [x] Implement Newton–Raphson solver
- [x] Implement Backward Euler (`BEuler`)
- [x] Test with arbitrary R, C
- [x] Plot V1, V2, V3, IE vs time
- [x] Sanity check: V3 is rectified

---

## 🗓️ Week 2 — Sensitivity Analysis
**Goal:** Sensitivities ready for Gauss-Newton

- [ ] Derive sensitivity equation for ∂X/∂R
- [ ] Derive sensitivity equation for ∂X/∂C
- [ ] Implement `getSensitivities()`
- [ ] Verify matrix dimensions `(T, 4)`
- [ ] Ensure sensitivities remain bounded
- [ ] (Optional) Plot sensitivities vs time
- [ ] Comment derivations for final report

---

## 🗓️ Week 3 — Gauss-Newton Parameter Estimation
**Goal:** Recover R and C from measurements

- [ ] Load `measurements.csv`
- [ ] Implement Gauss-Newton loop
- [ ] Compute residual vector
- [ ] Assemble Jacobian matrix `Jr`
- [ ] Update R and C using exponential rule
- [ ] Test convergence with good initial guess
- [ ] Test convergence with bad initial guess
- [ ] Plot cost vs iteration
- [ ] Plot V3(predicted) vs V3(measured)
- [ ] Analyze noise sensitivity

---

## 🗓️ Week 4 — Dataset Generation & Baseline ML
**Goal:** ML pipeline exists and runs

- [ ] Implement `create_dataset()`
- [ ] Choose R, C sampling strategy
- [ ] Generate clean simulation data
- [ ] Add noise to simulations
- [ ] Save dataset as `.pkl`
- [ ] Verify shapes: `X ∈ (N, T, 4)`, `Y ∈ (N, 2)`
- [ ] Train Linear / Ridge regression
- [ ] Train Random Forest regressor
- [ ] Evaluate prediction error on test set

---

## 🗓️ Week 5 — ML Optimization & Hybrid Method
**Goal:** Best ML model + ML-assisted Gauss-Newton

- [ ] Try SVR
- [ ] Try MLP (if needed)
- [ ] Perform feature selection (V3 only, etc.)
- [ ] Normalize / preprocess data
- [ ] Compare models (accuracy, speed, robustness)
- [ ] Select final ML model
- [ ] Use ML output as Gauss-Newton initial guess
- [ ] Compare hybrid vs pure Gauss-Newton
- [ ] Generate comparison plots

---

## 🗓️ Week 6 — Final Deliverables & Polish
**Goal:** Max points, zero stress

- [ ] Finalize all plots
- [ ] Write Jacobian derivation section
- [ ] Write sensitivity derivation section
- [ ] Add pseudo-code to report
- [ ] Write ML theory & comparison section
- [ ] Assign team contributions
- [ ] Format report in IEEE two-column style
- [ ] Clean and comment code
- [ ] Test code on clean environment
- [ ] Prepare slides (6 min + Q&A)

---

## ✅ Final Checklist
- [ ] Code runs without errors
- [ ] Results are reproducible
- [ ] All rubric items addressed
- [ ] Dataset + notebooks load correctly
- [ ] Ready to submit 🚀