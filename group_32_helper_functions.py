import numpy as np
from group_32_circuit_simulator import CircuitSimulator
import random as rd
import os
import pickle

def plot_data(x_test, tpoints):
    import matplotlib.pyplot as plt

    # Create the figure and the first axis (for Volts)
    _ ,ax1 = plt.subplots(figsize=(10, 6))

    # Plot V_1, V_2, and V_3 on the primary y-axis
    ax1.plot(tpoints, x_test[:, 0], label='$V_1$')
    ax1.plot(tpoints, x_test[:, 1], label='$V_2$', linestyle='--')
    ax1.plot(tpoints, x_test[:, 2], label='$V_3$', linestyle='--')

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Volt (V)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a twin axis for I_E (for mA)
    ax2 = ax1.twinx()
    # Scale I_E to mA
    ie_ma = x_test[:, 3] * 1000
    ax2.plot(tpoints, ie_ma, label='$I_E$', color='red', linestyle='--')
    ax2.set_ylabel("Current (mA)")

    # Calculate the ratio of the zero position relative to the range
    # We force the zero to be at the same proportional height on both axes
    def align_zeros(ax_ref, ax_target):
        ymin_ref, ymax_ref = ax_ref.get_ylim()
        rat = ymax_ref / (ymax_ref - ymin_ref)
        ymin_tar, ymax_tar = ax_target.get_ylim()
        if abs(ymin_tar) > abs(ymax_tar):
            new_ymax = ymin_tar * rat / (rat - 1)
            ax_target.set_ylim(ymin_tar, new_ymax)
        else:
            new_ymin = ymax_tar * (rat - 1) / rat
            ax_target.set_ylim(new_ymin, ymax_tar)

    # Apply the alignment
    align_zeros(ax1, ax2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("Voltage and Current vs. Time")
    plt.tight_layout()
    plt.show()
    
def create_dataset(num_samples, amplitude, f, delta_t, T, noise):
    """
    Build a supervised dataset from Backward Euler time-domain simulations.

    For each sample, R and C are drawn uniformly from the project ranges
    R in [1 Ohm, 2.5 kOhm] and C in [0.1 uF, 5 uF] (stored in ohms and farads).
    Rows of X are length-T trajectories with columns [V1, V2, V3, I_E].
    Rows of y are [R, C]. Pass noise=True for additive measurement noise
    on the simulated signals (as implemented in CircuitSimulator.BEuler).
    """
    x = []  # Simulated transient responses: shape will be (N, T, 4)
    y = []  # Ground-truth parameters: shape (N, 2)

    r_lo, r_hi = 1.0, 2500.0  # 2.5 kOhm
    c_lo, c_hi = 0.1e-6, 5e-6  # farads (0.1 uF .. 5 uF)

    for i in range(num_samples):
        R = float(np.random.uniform(r_lo, r_hi))
        C = float(np.random.uniform(c_lo, c_hi))

        mna = CircuitSimulator(amplitude, f, R, C)
        y.append([R, C])

        x_init = np.zeros((4,))
        transient, _ = mna.BEuler(x_init, delta_t, T, noise=noise)
        x.append(transient)

        if i % 100 == 0:
            print(f"Created {i + 1} samples")

    # Convert lists to NumPy arrays for easier manipulation in ML frameworks
    x = np.array(x)
    y = np.array(y)
    return x, y

def save_dataset(x, y):
    """
    Serializes the generated data and targets into a pickle file.
    """
    data_to_save = {
        'data': x,    # The time-series simulation results
        'target': y   # The corresponding R and C values
    }
    # Save to a specific directory using binary write mode
    with open(os.path.join('data', 'dataset.pkl'), 'wb') as f:
        pickle.dump(data_to_save, f)
