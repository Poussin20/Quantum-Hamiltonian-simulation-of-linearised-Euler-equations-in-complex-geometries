import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from qulacs.gate import X, Z, RX, RY, RZ, CNOT, merge, DenseMatrix,add
from qulacs.state import inner_product
import matplotlib.pyplot as plt
import numpy as np


# ----------------------
# Parameters
# ----------------------
n = 4                 # qubits per spatial dimension
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
    """
    Map (component c, x-index ix, y-index iy) to 1D index in [0, Ndof).
    c = 0: p, 1: u, 2: v, 3: dummy
    """
    return c + n_components * (ix + Nx * iy)

# ----------------------
# Build PDE operator A (df/dt = A f)
# ----------------------
A = np.zeros((Ndof, Ndof), dtype=np.float64)

def central_diff_x(c_from, c_to, coef):
    """
    Add coef * ∂_x(field_from) into time derivative of field_to using central differences.
    """
    for ix in range(Nx):
        for iy in range(Ny):
            row = idx(c_to, ix, iy)
            ip, im = ix + 1, ix - 1
            if 0 <= ip < Nx:
                A[row, idx(c_from, ip, iy)] += coef / (2 * l)
            if 0 <= im < Nx:
                A[row, idx(c_from, im, iy)] -= coef / (2 * l)

def central_diff_y(c_from, c_to, coef):
    """
    Add coef * ∂_y(field_from) into time derivative of field_to using central differences.
    """
    for ix in range(Nx):
        for iy in range(Ny):
            row = idx(c_to, ix, iy)
            jp, jm = iy + 1, iy - 1
            if 0 <= jp < Ny:
                A[row, idx(c_from, ix, jp)] += coef / (2 * l)
            if 0 <= jm < Ny:
                A[row, idx(c_from, ix, jm)] -= coef / (2 * l)

# LEE in conservative regime:
# ∂t p = - (∂x u + ∂y v) - ū ∂x p
central_diff_x(1, 0, -1.0)
central_diff_y(2, 0, -1.0)
central_diff_x(0, 0, -u_bar)

# ∂t u = -∂x p - ū ∂x u
central_diff_x(0, 1, -1.0)
central_diff_x(1, 1, -u_bar)

# ∂t v = -∂y p - ū ∂x v
central_diff_y(0, 2, -1.0)
central_diff_x(2, 2, -u_bar)

# ∂t dummy = -ū ∂x dummy
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
    """
    Extract pressure field p(x,y) as (Ny, Nx) array from full state f.
    """
    p = np.zeros((Ny, Nx), dtype=np.float64)
    for ix in range(Nx):
        for iy in range(Ny):
            p[iy, ix] = f[idx(0, ix, iy)]
    return p

def exact_solution(f0, T):
    """
    Exact: f(T) = exp(A T) f(0)
    """
    U = expm(A * T)
    return U @ f0

def fdm_evolution(f0, T):
    """
    Classical FDM (forward Euler): f_{n+1} = f_n + tau_fdm * A f_n
    """
    steps = int(round(T / tau_fdm))
    f = f0.copy()
    for _ in range(steps):
        f = f + tau_fdm * (A @ f)
    return f


# U_step = exp(A * tau_q) as a 2^ntot x 2^ntot unitary (same mapping as above)



# HERE 
# https://dojo.qulacs.org/en/latest/notebooks/4.2_trotter_decomposition.html link for the trotterization
# Just have to implement the good gates and the U_j 
# Carreful : multi-controlled gate might be a bit heavy
#
U_step = expm(A * tau_q).astype(np.complex128)
gate_step = DenseMatrix(list(range(ntot)), U_step)

def quantum_evolution_using_qulacs(f0, T):
    """
    1. Embed f0 into a quantum state |ψ0> on ntot qubits.
    2. Apply the unitary step U_step = exp(A τ_q) s = T/τ_q times.
    3. Read back the amplitudes and interpret them as the discretized fields.
    """
    # normalize f0 to get amplitudes
    norm = np.linalg.norm(f0)
    psi0 = f0.astype(np.complex128) / (norm if norm > 0 else 1.0)

    # load into Qulacs quantum state
    qs = QuantumState(ntot)
    qs.load(psi0)

    steps = int(round(T / tau_q))
    circuit = QuantumCircuit(ntot)
    for _ in range(steps):
        circuit.add_gate(gate_step)

    circuit.update_quantum_state(qs)

    # extract amplitudes and rescale back to PDE normalization
    vec = np.array(qs.get_vector()) * norm
    # vec[k] corresponds to classical index k (same mapping)
    return vec.real














# ----------------------
# Compute snapshots & plot
# ----------------------
fig, axes = plt.subplots(len(times), 3, figsize=(11, 3.5 * len(times)))
all_values = []

for row, T in enumerate(times):
    # compute all three solutions
    f_exact = exact_solution(f0, T)
    f_q     = quantum_evolution_using_qulacs(f0, T)
    f_fdm   = fdm_evolution(f0, T)

    all_values.extend(extract_pressure(f_exact).flatten())
    all_values.extend(extract_pressure(f_q).flatten())
    all_values.extend(extract_pressure(f_fdm).flatten())

    vmin = min(all_values)
    vmax = max(all_values)
    p_exact = extract_pressure(f_exact)
    p_q     = extract_pressure(f_q)
    p_fdm   = extract_pressure(f_fdm)

    data_list  = [p_q, p_exact, p_fdm]
    title_list = ["Quantum simulation", "Matrix exponential", "Classical FDM"]

    for col, (data, title) in enumerate(zip(data_list, title_list)):
        ax = axes[row, col] if len(times) > 1 else axes[col]
        im = ax.imshow(data, origin="lower", cmap="seismic", vmin=-0.1, vmax=0.1)
        ax.set_title(f"{title}\nT = {T}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
