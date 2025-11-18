from qulacs import QuantumState, QuantumCircuit, Observable, PauliOperator
from qulacs.gate import X, Z, RX, RY, RZ, CNOT, merge, DenseMatrix,add
from qulacs.state import inner_product
import matplotlib.pyplot as plt
import numpy as np


# In this case, we consider a system with six particles.
nqubits = 6
#  Time to simulate dynamics
t = 3.0
# Number of divisions of Trotter decomposition
M = 100
# Time increment range
delta = t/M

# Transverse magnetic field strength
h = 3.

# Prepare observable for all magnetizations.
magnetization_obs = Observable(nqubits)
for i in range(nqubits):
     magnetization_obs.add_operator(PauliOperator("Z "+str(i), 1.0))

# The initial state is |000000>
state_trotter = QuantumState(nqubits)
state_trotter.set_zero_state()
state_exact = QuantumState(nqubits)
state_exact.set_zero_state()

# Convert one Trotter decomposition ,e^{iZ_1Z_2*delta}*e^{iZ_2Z_3*delta}*...e^{iZ_nZ_1*delta} to a quantum gate
circuit_trotter_transIsing = QuantumCircuit(nqubits)
for i in range(nqubits):
    circuit_trotter_transIsing.add_CNOT_gate(i,(i+1)%(nqubits))
    circuit_trotter_transIsing.add_RZ_gate((i+1)%nqubits,2*delta) ## RZ(a)=exp(i*a/2*Z)
    circuit_trotter_transIsing.add_CNOT_gate(i,(i+1)%(nqubits))
    circuit_trotter_transIsing.add_RX_gate(i, 2*delta*h) ## RX(a)=exp(i*a/2*X)

# Diagonalize e^{-iHt} directly. To get the matrix representation of H, generate a gate and get its matrix
zz_matrix = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) ## matrix representation of Z_i*Z_{i+1}
hx_matrix = h*np.array( [ [0,1], [1,0] ] )
zz = DenseMatrix([0,1], zz_matrix) ## interaction between 0 and 1
hx = DenseMatrix(0, hx_matrix) ## transeverse magnetic field to 0 sight
## Add interactions and transverse magnetic fields at sight after 1 byqulacs.gate.add.
for i in range(1, nqubits):
    zz = add(zz, DenseMatrix([i,(i+1)%nqubits], zz_matrix))
    hx = add(hx, DenseMatrix(i, hx_matrix) )
## Final Hamiltonian
ham = add(zz, hx)
matrix = ham.get_matrix() #get matrix
eigenvalue, P = np.linalg.eigh(np.array(matrix)) #get eigenstate and eigenvector
## create e^{-i*H*delta} as a matrix
e_iHdelta = np.diag(np.exp(-1.0j*eigenvalue*delta))
e_iHdelta = np.dot(P, np.dot(e_iHdelta, P.T))
## convert to circuit
circuit_exact_transIsing = QuantumCircuit(nqubits)
circuit_exact_transIsing.add_dense_matrix_gate( np.arange(nqubits), e_iHdelta)

## A list that records time and magnetization
x = [i*delta for i in range(M+1)]
y_trotter = []
y_exact = []

#Calculate total magnetization at t=0
y_trotter.append( magnetization_obs.get_expectation_value(state_trotter) )
y_exact.append( magnetization_obs.get_expectation_value(state_exact) )

#Calculate total magnetization after t=0
for i in range(M):
    # Time evolution by delta=t/M
    circuit_trotter_transIsing.update_quantum_state(state_trotter)
    circuit_exact_transIsing.update_quantum_state(state_exact)
    # Calculate and record magnetization
    y_trotter.append( magnetization_obs.get_expectation_value(state_trotter) )
    y_exact.append( magnetization_obs.get_expectation_value(state_exact) )

#Drawing a graph
plt.xlabel("time")
plt.ylabel("Value of magnetization")
plt.title("Dynamics of transverse Ising model")
plt.plot(x, y_trotter, "-", label="Trotter")
plt.plot(x, y_exact, "-", label="exact")
plt.legend()
plt.show()