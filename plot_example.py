import numpy as np
from circuit_simulator import CircuitSimulator
from helper_functions import plot_data
import matplotlib.pyplot as plt


R = 2500            
C = 3e-6            
amplitude = 5       
frequency = 60      

delta_t = 1e-4
T = 0.05

mna = CircuitSimulator(amplitude, frequency, R, C)
x_init = np.zeros(4)
x_test, tpoints = mna.BEuler(x_init, delta_t, T, noise=False)

plot_data(x_test, tpoints)

# ==========================================
# Parameter Estimation (Gauss-Newton) and V3 comparison
# ==========================================
# Load true measurements 
x_true = np.loadtxt('measurements.csv', delimiter=',')

R_guess = 2500         # Initial guess for Resistor
C_guess = 3e-6          # Initial guess for Capacitor

# Initialize the Modified Nodal Analysis (MNA) object
mna_est = CircuitSimulator(amplitude, frequency, R_guess, C_guess)

# Run Gauss-Newton optimization 
R_pred, C_pred, cost = mna_est.GaussNewton(R_guess, C_guess, x_init, x_true, delta_t, T, max_iter=10)

print(f"Predicted resistor value:  {R_pred}.")
print(f"Predicted capacitor value: {C_pred}.")
print(f"Cost                     : {cost}.")

# Simulate with predicted R and C
mna_pred = CircuitSimulator(amplitude, frequency, R_pred, C_pred)
x_pred, tpoints_pred = mna_pred.BEuler(x_init, delta_t, T, noise=False)

print(f"True R: {R}, Predicted R: {R_pred}")
print(f"True C: {C}, Predicted C: {C_pred}")
# Plot V3 comparison
plt.figure(figsize=(10, 6))
plt.plot(tpoints, x_pred[:, 2], label='Predicted V3', color='blue')
plt.plot(tpoints, x_true[:, 2], label='True V3', color='red', linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Comparison of Predicted V3 and True V3")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
