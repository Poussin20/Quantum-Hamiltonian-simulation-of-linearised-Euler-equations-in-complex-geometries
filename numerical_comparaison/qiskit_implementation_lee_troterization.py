import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate


# ------------------- P_j(lambda) -------------------

def add_Pj(qc: QuantumCircuit, qubits, j: int, lam: float):
    target = qubits[j]
    qc.p(lam, target)


# ------------------- U_j(lambda) and its inverse -------------------

def add_Uj(qc: QuantumCircuit, qubits, j: int, lam: float):
    target = qubits[j]

    # H
    qc.h(target)

    # P_j(lambda)
    add_Pj(qc, qubits, j, lam)

    # CNOT(target, qubits[m]) for m = 0..j-1
    for m in range(j):
        qc.cx(target, qubits[m])


def add_Uj_dagger(qc: QuantumCircuit, qubits, j: int, lam: float):
    target = qubits[j]

    # inverse order of CNOTs
    for m in reversed(range(j)):
        qc.cx(target, qubits[m])

    # P_j(-lambda)
    add_Pj(qc, qubits, j, -lam)

    # H
    qc.h(target)


# ------------------- Multi-controlled RZ -------------------

def add_MCRZ(qc: QuantumCircuit, control_qubits, target_qubit: int, theta: float):
    if not control_qubits:
        qc.rz(theta, target_qubit)
    else:
        gate = RZGate(theta).control(len(control_qubits))
        qc.append(gate, control_qubits + [target_qubit])


# ------------------- Multi-controlled RZZ -------------------
# Using decomposition given in Fig 13 

def add_MCRZZ(qc: QuantumCircuit,
              control_qubits,
              qubit_a: int,
              qubit_b: int,
              theta: float):
    qc.x(qubit_a)

    controls = list(control_qubits) + [qubit_a]
    add_MCRZ(qc, controls, qubit_b, theta)

    qc.x(qubit_a)

    add_MCRZ(qc, controls, qubit_b, -theta)


# ------------------- Wxj and Wyj blocks -------------------

def add_Wxj(qc: QuantumCircuit,
            a1: int, a2: int,
            qx,
            j: int,
            tau: float,
            u_bar: float,
            rho_bar: float,
            l: float):
    lam = -np.pi / 2.0

    # H gate on a2
    qc.h(a2)
    # X gate on a1
    qc.x(a1)

    # Uj_dagger on qx starting at j
    add_Uj_dagger(qc, qx, j, lam)

    controls_zz = [a1] + [qx[m] for m in range(j)]

    theta_zz = (-1.0 * tau) / (rho_bar * l)
    # MCRZZ on (qx[j], a2) with given controls
    add_MCRZZ(qc, controls_zz, qx[j], a2, theta_zz)

    # Uj on qx starting at j
    add_Uj(qc, qx, j, lam)

    # H gate on a2
    qc.h(a2)
    # X gate on a1
    qc.x(a1)

    # Uj_dagger on qx starting at j
    add_Uj_dagger(qc, qx, j, lam)

    controls = [qx[m] for m in range(j)]

    theta_z = (-1.0 * u_bar * tau) / l
    if controls:
        add_MCRZ(qc, controls, qx[j], theta_z)
    else:
        qc.rz(theta_z, qx[j])

    # Uj on qx starting at j
    add_Uj(qc, qx, j, lam)


def add_Wyj(qc: QuantumCircuit,
            a1: int, a2: int,
            qy,
            j: int,
            tau: float,
            rho_bar: float,
            l: float):
    lam = -np.pi / 2.0

    qc.x(a2)
    qc.h(a1)

    add_Uj_dagger(qc, qy, j, lam)

    controls_zz = [a2] + [qy[m] for m in range(j)]

    theta_zz = (-1.0 * tau) / (rho_bar * l)
    add_MCRZZ(qc, controls_zz, qy[j], a1, theta_zz)

    add_Uj(qc, qy, j, lam)

    qc.x(a2)
    qc.h(a1)


# ------------------- Qx(tau), Qy(tau) layers -------------------

def add_Qx_layer(qc: QuantumCircuit,
                 a1: int, a2: int,
                 qx,
                 tau: float,
                 u_bar: float,
                 rho_bar: float,
                 l: float):
    n = len(qx)
    for j in range(n):
        add_Wxj(qc, a1, a2, qx, j, tau, u_bar, rho_bar, l)


def add_Qy_layer(qc: QuantumCircuit,
                 a1: int, a2: int,
                 qy,
                 tau: float,
                 rho_bar: float,
                 l: float):
    n = len(qy)
    for j in range(n):
        add_Wyj(qc, a1, a2, qy, j, tau, rho_bar, l)


# ------------------- One Lee Trotter step -------------------

def build_lee_trotter_step(n: int,
                           tau: float,
                           u_bar: float,
                           rho_bar: float,
                           l: float) -> QuantumCircuit:
    n_qubits = 2 + 2 * n
    qc = QuantumCircuit(n_qubits)

    a1, a2 = 0, 1
    qx = list(range(2, 2 + n))
    qy = list(range(2 + n, 2 + 2 * n))

    add_Qx_layer(qc, a1, a2, qx, tau, u_bar, rho_bar, l)
    add_Qy_layer(qc, a1, a2, qy, tau, rho_bar, l)

    return qc



def build_lee_trotter_evolution_qiskit(n: int,
                                tau: float,
                                steps: int,
                                u_bar: float,
                                rho_bar: float,
                                l: float) -> QuantumCircuit:
    base = build_lee_trotter_step(n, tau, u_bar, rho_bar, l)
    qc = QuantumCircuit(base.num_qubits)

    for _ in range(steps):
        qc.compose(base, inplace=True)

    return qc
