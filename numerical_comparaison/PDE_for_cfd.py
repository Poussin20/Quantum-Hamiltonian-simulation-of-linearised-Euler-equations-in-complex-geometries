import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from qulacs.gate import X, Z, RX, RY, RZ, CNOT, merge, DenseMatrix,add
from qulacs.state import inner_product
import matplotlib.pyplot as plt
import numpy as np

import lee_trotter

from qiskit.quantum_info import Statevector
import qiskit_implementation_lee_troterization


# ----------------------
# Parameters
# ----------------------
n = 5                # qubits per spatial dimension
Nx = 2**n
Ny = 2**n

n_components = 4      # (p, u, v, dummy)
l = 0.25              # lattice spacing
rho_bar = 1.0
c = 1.0
u_bar = -1.0          # background flow in x

assert abs(c - 1.0 / rho_bar) < 1e-12  # conservative regime

Ndof = n_components * Nx * Ny        # classical DOFs
ntot = 2 + 2*n                       # qubits: a1, a2, x-reg (n), y-reg (n)
assert 2**ntot == Ndof               # amplitude-encoding dimension matches

tau_q = 0.05                         # quantum time step
tau_fdm = 0.005                      # FDM step
times = [0.0, 1.5, 3.0] 

print(f"Grid: {Nx} x {Ny}, DOF = {Ndof}")

# ----------------------
# Indexing helper
# ----------------------
def idx(c, ix, iy):
    return c + n_components * (ix + Nx * iy)

def basis_index(a1, a2, ix, iy, n):
    idx = 0
    # ancillas
    idx |= (a1 & 1) << 0
    idx |= (a2 & 1) << 1

    # x register
    for b in range(n):
        bit = (ix >> b) & 1
        idx |= bit << (2 + b)

    # y register
    for b in range(n):
        bit = (iy >> b) & 1
        idx |= bit << (2 + n + b)

    return idx


# ----------------------
# Build PDE operator A (df/dt = A f)
# ----------------------
A = np.zeros((Ndof, Ndof), dtype=np.float64)


def central_diff_x(c_from, c_to, coef):
    for ix in range(Nx):
        for iy in range(Ny):
            row = idx(c_to, ix, iy)
            ip, im = ix + 1, ix - 1
            if 0 <= ip < Nx:
                A[row, idx(c_from, ip, iy)] += coef / (2 * l)
            if 0 <= im < Nx:
                A[row, idx(c_from, im, iy)] -= coef / (2 * l)

def central_diff_y(c_from, c_to, coef):
    for ix in range(Nx):
        for iy in range(Ny):
            row = idx(c_to, ix, iy)
            jp, jm = iy + 1, iy - 1
            if 0 <= jp < Ny:
                A[row, idx(c_from, ix, jp)] += coef / (2 * l)
            if 0 <= jm < Ny:
                A[row, idx(c_from, ix, jm)] -= coef / (2 * l)

# LEE in conservative regime:
central_diff_x(1, 0, -1.0)
central_diff_y(2, 0, -1.0)
central_diff_x(0, 0, -u_bar)

central_diff_x(0, 1, -1.0)
central_diff_x(1, 1, -u_bar)

central_diff_y(0, 2, -1.0)
central_diff_x(2, 2, -u_bar)

central_diff_x(3, 3, -u_bar)

# ----------------------
# Initial condition: 2x2 high-pressure block in center
# ----------------------
f0 = np.zeros(Ndof, dtype=np.float64)

cx = Nx // 2
cy = Ny // 2
for dx in [0, 1]:
    for dy in [0, 1]:
        ix = cx + dx - 1
        iy = cy + dy - 1
        if 0 <= ix < Nx and 0 <= iy < Ny:
            f0[idx(0, ix, iy)] = 0.5  # pressure = 0.5 in the block

# ----------------------
# Helpers: extract p, evolution methods
# ----------------------
def extract_pressure(f):
    p = np.zeros((Ny, Nx), dtype=np.float64)
    for ix in range(Nx):
        for iy in range(Ny):
            p[iy, ix] = f[idx(0, ix, iy)]
    return p

def extract_pressure_from_psi(psi, n, Nx, Ny, energy_norm=1.0):
    p = np.zeros((Ny, Nx), dtype=np.float64)
    for ix in range(Nx):
        for iy in range(Ny):
            idx_q = basis_index(a1=0, a2=0, ix=ix, iy=iy, n=n)
            amp = psi[idx_q]
            p[iy, ix] = energy_norm * amp.real
    return p

def exact_solution(f0, T):
    U = expm(A * T)
    return U @ f0

def fdm_evolution(f0, T):
    steps = int(round(T / tau_fdm))
    f = f0.copy()
    for _ in range(steps):
        f = f + tau_fdm * (A @ f)
    return f


###################################################################################################
#######################################____#####____###############################################
###########################################\###/###################################################
############################################-Q-####################################################


# ----------------------
# Compute the quantum evolution using the LEE trotter step
#  Trotter step V(tau) ≈ Q_y(Tau) Q_x(Tau)
# ----------------------
def quantum_evolution_for_lee(f0, T):
    ntot = 2 + 2 * n
    dim = 2**ntot

    assert f0.size == dim

    #psi0 = lee_trotter.build_lee_initial_state(n, amplitude=0.5)
    steps = int(round(T / tau_q))
    #psiT = lee_trotter.evolve_lee(n, tau_q, steps, u_bar, rho_bar, l, psi0)
    psiT = lee_trotter.evolve_lee_default_ic(n, tau_q, steps, u_bar, rho_bar, l)
    return np.array(psiT)


def quantum_evolution_for_lee_qiskit(f0, T):
    ntot = 2 + 2 * n
    dim = 2**ntot
    assert f0.size == dim 

    psi0 = np.zeros(dim, dtype=complex)

    cx = Nx // 2
    cy = Ny // 2
    for dx in [0, 1]:
        for dy in [0, 1]:
            ix = cx + dx - 1
            iy = cy + dy - 1
            if 0 <= ix < Nx and 0 <= iy < Ny:
                idx_q = basis_index(a1=0, a2=0, ix=ix, iy=iy, n=n)
                psi0[idx_q] = 0.5

    steps = int(round(T / tau_q))
    qc = qiskit_implementation_lee_troterization.build_lee_trotter_evolution_qiskit(
        n=n,
        tau=tau_q,
        steps=steps,
        u_bar=u_bar,
        rho_bar=rho_bar,
        l=l,
    )

    state = Statevector(psi0)
    state_T = state.evolve(qc)

    return np.asarray(state_T.data)


# ----------------------
# Compute snapshots & plot
# ----------------------
fig, axes = plt.subplots(len(times), 3, figsize=(11, 3.5 * len(times)))
all_values = []

fig.suptitle(
    f"l = {l:.3f},  rho_bar= {rho_bar:.3f},  c = {c:.3f},  u_bar = {u_bar:.3f},  "
    f"tau_q = {tau_q:.3f},  tau_fdm = {tau_fdm:.3f}",
    fontsize=14,
    y=0.98
)

for row, T in enumerate(times):

    # compute all three solutions
    f_exact = exact_solution(f0, T)
    f_q_qiskit     = quantum_evolution_for_lee_qiskit(f0, T)
    #f_q     = quantum_evolution_for_lee(f0, T)
    f_fdm   = fdm_evolution(f0, T)

    all_values.extend(extract_pressure(f_exact).flatten())
    all_values.extend(extract_pressure(f_q_qiskit).flatten())
    #all_values.extend(extract_pressure(f_q).flatten())
    all_values.extend(extract_pressure(f_fdm).flatten())

    vmin = min(all_values)
    vmax = max(all_values)
    p_exact = extract_pressure(f_exact)
    p_q_qiskit     = extract_pressure_from_psi(f_q_qiskit, n, Nx, Ny, energy_norm=1.0)
    #p_q     = extract_pressure_from_psi(f_q, n, Nx, Ny, energy_norm=1.0)
    p_fdm   = extract_pressure(f_fdm)

    data_list  = [p_q_qiskit, p_exact, p_fdm]
    title_list = ["Quantum simulation in Qiskit", "Matrix exponential", "Classical FDM"]

    for col, (data, title) in enumerate(zip(data_list, title_list)):
        ax = axes[row, col] if len(times) > 1 else axes[col]
        im = ax.imshow(data, origin="lower", cmap="seismic", vmin=-0.1, vmax=0.1)
        ax.set_title(f"{title}\nT = {T}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

