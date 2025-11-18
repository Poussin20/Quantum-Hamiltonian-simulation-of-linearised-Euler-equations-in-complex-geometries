from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from qulacs.gate import X, Z, RX, RY, RZ, CNOT, merge, DenseMatrix,add
from qulacs.state import inner_product
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------
# U_j(lambda) from Eq. (13) acting on the x or y register
# qubits: list of physical indices [q1,...,qn]
# j     : 0-based index for j
# lambda: phase angle
# ----------------------------------------------------------
def add_Uj(circ, qubits, j, lam):
    target = qubits[j]
    # Hj
    circ.add_H_gate(target)
    # Pj(lambda) ~ RZ(-lambda) up to global phase
    circ.add_RZ_gate(target, -lam)
    # CNOT_{control=j,target=m}
    for m in range(j):
        circ.add_CNOT_gate(target, qubits[m])

def add_Uj_dagger(circ, qubits, j, lam):
    target = qubits[j]
    # inverse of U_j(lambda):
    for m in reversed(range(j)):
        circ.add_CNOT_gate(target, qubits[m])
    circ.add_RZ_gate(target, +lam)   # inverse of RZ(-lam)
    circ.add_H_gate(target)          # Hadamard is self-inverse


# ----------------------------------------------------------
# Multi-controlled RZ on target, controls in control_qubits
# Implements exp(-i theta Z_target / 2) only when all controls=1.
# ----------------------------------------------------------
def add_MCRZ(circ, control_qubits, target_qubit, theta):
    all_qubits = control_qubits + [target_qubit]
    k = len(all_qubits)
    dim = 2 ** k
    mat = np.eye(dim, dtype=complex)

    # build diagonal matrix
    for b in range(dim):
        # bit layout: [controls..., target] with LSB at position 0
        ctrl_ok = True
        for ci, cq in enumerate(control_qubits):
            bit = (b >> ci) & 1
            if bit != 1:
                ctrl_ok = False
                break
        t_bit = (b >> (len(control_qubits))) & 1

        if ctrl_ok:
            # RZ(theta) = diag(exp(-itheta/2), exp(+itheta/2))
            phase = np.exp(-1j * theta / 2) if t_bit == 0 else np.exp(+1j * theta / 2)
            mat[b, b] = phase
        else:
            mat[b, b] = 1.0

    gate = DenseMatrix(all_qubits, mat)
    circ.add_gate(gate)


# ----------------------------------------------------------
# Multi-controlled RZZ on (qA, qB), controlled on control_qubits.
# Implements exp(-i theta Z_qA Z_qB / 2) when all controls = 1.
# ----------------------------------------------------------
def add_MCRZZ(circ, control_qubits, qubit_a, qubit_b, theta):
    all_qubits = control_qubits + [qubit_a, qubit_b]
    k = len(all_qubits)
    dim = 2 ** k
    mat = np.eye(dim, dtype=complex)

    for b in range(dim):
        ctrl_ok = True
        for ci in range(len(control_qubits)):
            bit = (b >> ci) & 1
            if bit != 1:
                ctrl_ok = False
                break

        # indices of the last two bits in this local ordering
        a_bit = (b >> len(control_qubits)) & 1
        b_bit = (b >> (len(control_qubits) + 1)) & 1

        if ctrl_ok:
            # ZxZ eigenvalues: +1 for 00,11 ; -1 for 01,10
            zz = 1 if (a_bit == b_bit) else -1
            # RZZ(theta) = exp(-i theta ZxZ / 2)
            phase = np.exp(-1j * theta * zz / 2)
            mat[b, b] = phase
        else:
            mat[b, b] = 1.0

    gate = DenseMatrix(all_qubits, mat)
    circ.add_gate(gate)


# ----------------------------------------------------------
# Wx,j block (x-direction), Eq. (28)
# a1, a2              : ancilla indices
# qx                  : list [qx1, ..., qxn]
# j                   : 0-based index
# tau, u_bar, rho_bar : physical params
# l                   : grid spacing
# ----------------------------------------------------------
def add_Wxj(circ, a1, a2, qx, j, tau, u_bar, rho_bar, l):
    lam = -np.pi / 2

    # ---- First factor: U_j(-π/2) MCRZ(-u_barTau/l) U_j(-π/2)(dagger) ----
    add_Uj(circ, qx, j, lam)
    # controls: qx[0..j-1], target: qx[j]
    controls = [qx[m] for m in range(j)]
    theta_z = -u_bar * tau / l
    if j > 0:
        add_MCRZ(circ, controls, qx[j], theta_z)
    else:
        # for j = 0, no controls -> plain RZ
        circ.add_RZ_gate(qx[j], theta_z)
    add_Uj_dagger(circ, qx, j, lam)

    # ---- Second factor: (H_a1 x X_a2 x U_j) MCRZZ(...) (U_j(dagger) x X_a1 x H_a2) ----
    circ.add_H_gate(a1)
    circ.add_X_gate(a2)
    add_Uj(circ, qx, j, lam)

    # MCRZZ_{a1,1,...,j-1}^{a2,j}(-Tau/(rho_bar * l)
    controls_zz = [a1] + [qx[m] for m in range(j)]   # a1, qx1..qx,j-1
    theta_zz = -tau / (rho_bar * l)
    add_MCRZZ(circ, controls_zz, a2, qx[j], theta_zz)

    add_Uj_dagger(circ, qx, j, lam)
    circ.add_X_gate(a1)
    circ.add_H_gate(a2)

# ----------------------------------------------------------
# Wy,j block (y-direction), Eq. (28)
# a1, a2              : ancilla indices
# qy                  : list [qy1, ..., qyn]
# j                   : 0-based
# ----------------------------------------------------------
def add_Wyj(circ, a1, a2, qy, j, tau, rho_bar, l):
    lam = -np.pi / 2

    # (H_a1 x X_a2 x U_j(-π/2)) MCRZZ_{a2,1,...,j-1}^{a1,j}(-Tau/(rho_bar * l))
    #   (U_j(-π/2)(dagger) x H_a1 x X_a2)
    circ.add_H_gate(a1)
    circ.add_X_gate(a2)
    add_Uj(circ, qy, j, lam)

    controls_zz = [a2] + [qy[m] for m in range(j)]   # a2, qy1..qy,j-1
    theta_zz = -tau / (rho_bar * l)
    add_MCRZZ(circ, controls_zz, a1, qy[j], theta_zz)

    add_Uj_dagger(circ, qy, j, lam)
    circ.add_H_gate(a1)
    circ.add_X_gate(a2)



# ----------------------------------------------------------
# Build Qx(Tau)
# ----------------------------------------------------------
def add_Qx_layer(circ, a1, a2, qx, tau, u_bar, rho_bar, l):
    n = len(qx)
    for j in range(n):  # j = 0..n-1
        add_Wxj(circ, a1, a2, qx, j, tau, u_bar, rho_bar, l)

def add_Qy_layer(circ, a1, a2, qy, tau, rho_bar, l):
    n = len(qy)
    for j in range(n):
        add_Wyj(circ, a1, a2, qy, j, tau, rho_bar, l)

# ----------------------------------------------------------
# Build one Trotter step circuit for the LEE Hamiltonian
# Eq. 26-27 in the paper
# ----------------------------------------------------------
def build_lee_trotter_step(n, tau, u_bar, rho_bar, l):
    n_qubits = 2 + 2 * n  # 2 ancilla + n x-qubits + n y-qubits
    circ = QuantumCircuit(n_qubits)

    a1, a2 = 0, 1
    qx = [2 + i for i in range(n)]
    qy = [2 + n + i for i in range(n)]

    # Eq. (26): V(Tau) ~= e^{-iHyTau} e^{-iHxTau} = Qy(Tau) Qx(Tau)
    add_Qy_layer(circ, a1, a2, qy, tau, rho_bar, l)
    add_Qx_layer(circ, a1, a2, qx, tau, u_bar, rho_bar, l)

    return circ
