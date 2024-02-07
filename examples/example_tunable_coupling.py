from main import QSystem
from math import pi
import numpy as np
from qutip import steadystate, expect, Qobj
import matplotlib.pyplot as plt

# let's investigate 8q system
# below you can find required constants
w = 6.5 * 2 * pi  # frequencies of qubits, GHz
wc_pos = 6 * 2 * pi
wc_neg = 7 * 2 * pi
g1 = 80e-3 * 2 * pi  # coupling coefficient, GHz
g2 = 80e-3 * 2 * pi
g12 = 3e-3 * 2 * pi
gamma = 16e-3  # decay rates, GHz
Gamma = gamma

# In the next block, we will compute transmission spectre of the system using qutip.steadystate
drive_frequencies = np.linspace(6.484, 6.509, 100) * 2 * pi
drive_amplitude = 0.1e-2 * 2 * pi
coupler_frequencies = np.linspace(8, 10, 100) * 2 * pi
sparam_all = []
# effective coupling coefficient
g_eff = []
for wc in coupler_frequencies:
    print(wc / (2 * pi))
    sparam = []
    for w_drive in drive_frequencies:
        frequencies = [[0, w], [1, wc], [2, w]]
        couplings = [[1, 0, g1], [2, 1, g2], [0, 2, g12]]
        n_q = len(frequencies)
        single_photon_states = [[0 for _ in range(n_q)], ] + [[1 if j == i else 0 for j in range(n_q)] for i in
                                                              range(n_q)]  # basis states
        system = QSystem(frequencies, couplings, single_photon_states)  # initialization of the system
        H_full = system.H_matrix() + Qobj(
            np.array([[-w_drive if (i == j and i != 0) else 0 for j in range(len(single_photon_states))] for i in
                      range(len(single_photon_states))])) + drive_amplitude / 2 * (
                         system.annihilation_operator_matrix(0) + system.annihilation_operator_matrix(0).dag())
        final_state = steadystate(H_full, [np.sqrt(gamma) * system.annihilation_operator_matrix(0),
                                           np.sqrt(gamma) * system.annihilation_operator_matrix(2)],
                                  use_precond=True, maxiter=1000, tol=1e-12,
                                  return_info=True)
        s = 2 * Gamma / (1j * drive_amplitude) * expect(Qobj(system.annihilation_operator_matrix(2)).dag(),
                                                        final_state[0])
        sparam.append(s)
    # detection of the moment when coefficient g_eff becomes bigger than zero
    states = system.H_matrix().eigenstates()
    for i in range(len(states[1])):
        if states[1][i][1].real < 0 and states[1][i][3].real > 0:
            antisymmetric_index = i
        elif states[1][i][1].real > 0 and states[1][i][3].real > 0:
            symmetric_index = i
    g_eff.append((states[0][symmetric_index] - states[0][antisymmetric_index]) / 2)
    sparam_all.append(sparam)

# visualization block
# heatmap: x-axis is driving frequency, y-axis is coupler frequency and color is absolute value of the S-parameter
plt.imshow(np.abs(np.array(sparam_all)), extent=[0, 100, 0, 100], cmap='hot')
plt.yticks(np.arange(len(coupler_frequencies))[::-4],
           labels=list(map(lambda x: str(round(x / (2 * pi), 3)), coupler_frequencies[::4])), rotation=45, ha="right",
           rotation_mode="anchor")
plt.xticks(np.arange(len(drive_frequencies))[::4],
           labels=list(map(lambda x: str(round(x / (2 * pi), 3)), drive_frequencies[::4])), rotation=45, ha="right",
           rotation_mode="anchor")
plt.xlabel("$\omega_{drive}$, GHz", fontsize=14)
plt.ylabel("$\omega_c$, GHz", fontsize=14)
plt.colorbar(label="ABS[S]")
plt.show()

# The comparison between theoretical dependence, got with Schrieffer–Wolff transformation, with data numerical
# calculations, got from numerical calculations of the original hamiltonian
plt.plot(coupler_frequencies / (2 * pi), np.array(g_eff) * 1000, "rx", markersize="5")
plt.plot(coupler_frequencies / (2 * pi), g1 * g2 / (w - coupler_frequencies) * 1000 + g12 * 1000, color="orange")
plt.ylabel("$g_{eff}$, MHz", fontsize=14)
plt.xlabel("$\omega_c$, GHz", fontsize=14)
plt.show()
