#include <vector>
#include <complex>
#include <algorithm>

#include <Eigen/Core>
#include <cppsim/state.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/type.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace py = pybind11;
using namespace gate;
using UINT = unsigned int;
using CPPCTYPE = std::complex<double>;

// ------------------- U_j(lambda) and its inverse -------------------

void add_Uj(QuantumCircuit& circ,
            const std::vector<UINT>& qubits,
            UINT j,
            double lambda) {
    UINT target = qubits[j];

    // H
    circ.add_gate(H(target));

    // P_j(lambda) ~ RZ(lambda) up to global phase (paper Eq. 13)
    circ.add_gate(RZ(target, lambda));

    for (UINT m = 0; m < j; ++m) {
        circ.add_gate(CNOT(target, qubits[m]));
    }
}

void add_Uj_dagger(QuantumCircuit& circ,
                   const std::vector<UINT>& qubits,
                   UINT j,
                   double lambda) {
    UINT target = qubits[j];

    // inverse order and lambda
    for (int m = (int)j - 1; m >= 0; --m) {
        circ.add_gate(CNOT(target, qubits[(UINT)m]));
    }
    circ.add_gate(RZ(target, -lambda));
    circ.add_gate(H(target));
}

// ------------------- Multi-controlled RZ -------------------

void add_MCRZ(QuantumCircuit& circ,
              const std::vector<UINT>& control_qubits,
              UINT target_qubit,
              double theta) {
    std::vector<UINT> all_qubits = control_qubits;
    all_qubits.push_back(target_qubit);

    const UINT k   = (UINT)all_qubits.size();
    const UINT dim = 1u << k;

    ComplexVector diag(dim);
    diag.setOnes();

    for (UINT b = 0; b < dim; ++b) {
        bool ctrl_ok = true;
        for (UINT ci = 0; ci < control_qubits.size(); ++ci) {
            UINT bit = (b >> ci) & 1u;
            if (bit != 1u) { ctrl_ok = false; break; }
        }
        if (!ctrl_ok) continue;

        UINT t_bit = (b >> control_qubits.size()) & 1u;

        CPPCTYPE phase = (t_bit == 0u)
            ? std::exp(CPPCTYPE(0.0, -theta/2.0))
            : std::exp(CPPCTYPE(0.0, +theta/2.0));

        diag[b] = phase;
    }

    auto gate = gate::DiagonalMatrix(all_qubits, diag);
    circ.add_gate(gate);
}

// ------------------- Multi-controlled RZZ -------------------
// Implements exp(-i theta Z_a Z_b / 2) when all control_qubits = 1.
// Using decomposition: RZZ(a,b,theta) = CNOT(a,b) * RZ_b(theta) * CNOT(a,b)

void add_MCRZZ(QuantumCircuit& circ,
               const std::vector<UINT>& control_qubits,
               UINT qubit_a,
               UINT qubit_b,
               double theta) {
    circ.add_gate(CNOT(qubit_a, qubit_b));

    std::vector<UINT> controls = control_qubits;
    controls.push_back(qubit_a);

    add_MCRZ(circ, controls, qubit_b, theta);

    circ.add_gate(CNOT(qubit_a, qubit_b));
}

// ------------------- Wxj and Wyj blocks -------------------

void add_Wxj(QuantumCircuit& circ,
             UINT a1, UINT a2,
             const std::vector<UINT>& qx,
             UINT j,
             double tau,
             double u_bar,
             double rho_bar,
             double l) {
    double lambda = -M_PI / 2.0;

    // ---- First factor: U_j(-Pi/2) MCRZ(-u_bar tau / l) U_j(-Pi/2)(dagger) ----
    add_Uj(circ, qx, j, lambda);
    std::vector<UINT> controls;
    for (UINT m = 0; m < j; ++m) controls.push_back(qx[m]);

    double theta_z = -u_bar * tau / l;
    if (!controls.empty()) {
        add_MCRZ(circ, controls, qx[j], theta_z);
    } else {
        circ.add_gate(RZ(qx[j], theta_z));
    }
    add_Uj_dagger(circ, qx, j, lambda);

    // ---- Second factor: (H_a1 (x) X_a2 (x) U_j) MCRZZ(...) (U_j)(dagger) (x) X_a1 (x) H_a2) ----
    circ.add_gate(H(a1));
    circ.add_gate(X(a2));
    add_Uj(circ, qx, j, lambda);

    // controls for RZZ: a1 and qx[0..j-1]
    std::vector<UINT> controls_zz;
    controls_zz.push_back(a1);
    for (UINT m = 0; m < j; ++m) controls_zz.push_back(qx[m]);

    double theta_zz = -tau / (rho_bar * l);
    add_MCRZZ(circ, controls_zz, a2, qx[j], theta_zz);

    add_Uj_dagger(circ, qx, j, lambda);
    circ.add_gate(X(a1));
    circ.add_gate(H(a2));
}

void add_Wyj(QuantumCircuit& circ,
             UINT a1, UINT a2,
             const std::vector<UINT>& qy,
             UINT j,
             double tau,
             double rho_bar,
             double l) {
    double lambda = -M_PI / 2.0;

    circ.add_gate(H(a1));
    circ.add_gate(X(a2));
    add_Uj(circ, qy, j, lambda);

    // controls: a2 and qy[0..j-1]
    std::vector<UINT> controls_zz;
    controls_zz.push_back(a2);
    for (UINT m = 0; m < j; ++m) controls_zz.push_back(qy[m]);

    double theta_zz = -tau / (rho_bar * l);
    add_MCRZZ(circ, controls_zz, a1, qy[j], theta_zz);

    add_Uj_dagger(circ, qy, j, lambda);
    circ.add_gate(H(a1));
    circ.add_gate(X(a2));
}

// ------------------- Qx(tau), Qy(tau) layers and 1 step V(tau) -------------------

void add_Qx_layer(QuantumCircuit& circ,
                  UINT a1, UINT a2,
                  const std::vector<UINT>& qx,
                  double tau,
                  double u_bar,
                  double rho_bar,
                  double l) {
    UINT n = qx.size();
    for (UINT j = 0; j < n; ++j) {
        add_Wxj(circ, a1, a2, qx, j, tau, u_bar, rho_bar, l);
    }
}

void add_Qy_layer(QuantumCircuit& circ,
                  UINT a1, UINT a2,
                  const std::vector<UINT>& qy,
                  double tau,
                  double rho_bar,
                  double l) {
    UINT n = qy.size();
    for (UINT j = 0; j < n; ++j) {
        add_Wyj(circ, a1, a2, qy, j, tau, rho_bar, l);
    }
}

void build_lee_trotter_step(QuantumCircuit& circ,
                            int n,
                            double tau,
                            double u_bar,
                            double rho_bar,
                            double l) {
    UINT a1 = 1, a2 = 0;

    std::vector<UINT> qx(n), qy(n);
    for (int i = 0; i < n; ++i) {
        qx[i] = 2 + i;
        qy[i] = 2 + n + i;
    }

    add_Qx_layer(circ, a1, a2, qx, tau, u_bar, rho_bar, l);
    add_Qy_layer(circ, a1, a2, qy, tau, rho_bar, l);
}

// ------------------- Python-visible wrapper: evolve state -------------------

py::array_t<CPPCTYPE> evolve_lee(int n,
                                 double tau,
                                 int steps,
                                 double u_bar,
                                 double rho_bar,
                                 double l,
                       py::array_t<CPPCTYPE, py::array::c_style | py::array::forcecast> psi0) {
    int n_qubits = 2 + 2 * n;
    const std::size_t dim = (std::size_t)1 << n_qubits;

    if (psi0.size() != (py::ssize_t)dim) {
        throw std::runtime_error("psi0 has wrong length for given n.");
    }

    // Build one Trotter step circuit
    QuantumCircuit circ(n_qubits);
    build_lee_trotter_step(circ, n, tau, u_bar, rho_bar, l);

    // Create state and load psi0
    QuantumState state(n_qubits);
    auto buf = psi0.request();
    auto *in_ptr = static_cast<CPPCTYPE*>(buf.ptr);

    // Copy into Qulacs' internal buffer
    CPPCTYPE* data = const_cast<CPPCTYPE*>(state.data_cpp());
    std::copy(in_ptr, in_ptr + dim, data);

    // Apply 'steps' times
    for (int s = 0; s < steps; ++s) {
        circ.update_quantum_state(&state);
    }

    // Export result back to numpy
    py::array_t<CPPCTYPE> out(dim);
    auto buf_out = out.request();
    auto *out_ptr = static_cast<CPPCTYPE*>(buf_out.ptr);
    const CPPCTYPE* final_data = state.data_cpp();
    std::copy(final_data, final_data + dim, out_ptr);

    return out;
}

// ------------------- pybind11 module -------------------

PYBIND11_MODULE(lee_trotter, m) {
    m.doc() = "LEE Trotter step using Qulacs";

    m.def("evolve_lee", &evolve_lee,
          py::arg("n"),
          py::arg("tau"),
          py::arg("steps"),
          py::arg("u_bar"),
          py::arg("rho_bar"),
          py::arg("l"),
          py::arg("psi0"),
          "Evolve psi0 by 'steps', Trotter steps of the LEE Hamiltonian.\n"
          "Returns the final state vector as a complex numpy array.");
}
