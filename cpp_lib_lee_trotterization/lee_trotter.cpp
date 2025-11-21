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

inline std::size_t encode_basis_index(
    UINT comp,
    UINT ix, UINT iy,
    UINT a1, UINT a2,
    const std::vector<UINT>& qx,
    const std::vector<UINT>& qy
) {
    std::size_t idx = 0;

    // ancilla bits: a1 is LSB, a2 is MSB of 'comp'
    UINT a1_val = comp & 1u;
    UINT a2_val = (comp >> 1) & 1u;

    idx |= (std::size_t(a1_val) << a1);
    idx |= (std::size_t(a2_val) << a2);

    // x register
    for (UINT b = 0; b < qx.size(); ++b) {
        UINT bit = (ix >> b) & 1u;
        idx |= (std::size_t(bit) << qx[b]);
    }

    // y register
    for (UINT b = 0; b < qy.size(); ++b) {
        UINT bit = (iy >> b) & 1u;
        idx |= (std::size_t(bit) << qy[b]);
    }

    return idx;
}


// ------------------- P_j(lambda) -------------------

void add_Pj(QuantumCircuit& circ,
            const std::vector<UINT>& qubits,
            UINT j,
            double lambda) {
    CPPCTYPE im(0, 1);

    ComplexMatrix mat(2, 2);
    mat(0, 0) = 1;
    mat(0, 1) = 0;
    mat(1, 0) = 0;
    mat(1, 1) = exp(im * lambda);
    circ.add_gate(gate::DenseMatrix(qubits[j], mat));
}

// ------------------- U_j(lambda) and its inverse -------------------

void add_Uj(QuantumCircuit& circ,
            const std::vector<UINT>& qubits,
            UINT j,
            double lambda) {
    UINT target = qubits[j];

    // H
    circ.add_gate(H(target));
    
    add_Pj(circ, qubits, j, lambda);

    for (UINT m = 0; m < j; ++m) {
        circ.add_gate(CNOT(target, qubits[(UINT)m]));
    }
}

void add_Uj_dagger(QuantumCircuit& circ,
                   const std::vector<UINT>& qubits,
                   UINT j,
                   double lambda) {
    UINT target = qubits[j];

    // inverse order

    for (int m = (int)j - 1; m >= 0; --m) {
        circ.add_gate(CNOT(target, qubits[(UINT)m]));
    }

    add_Pj(circ, qubits, j, -lambda);

    circ.add_gate(H(target));
}

// ------------------- Multi-controlled RZ -------------------

void add_MCRZ(QuantumCircuit& circ,
              const std::vector<UINT>& control_qubits,
              UINT target_qubit,
              double theta) {    
    CPPCTYPE p0 = std::exp(CPPCTYPE(0.0, -theta/2.0));
    CPPCTYPE p1 = std::exp(CPPCTYPE(0.0, +theta/2.0));

    ComplexMatrix mat(2, 2);
    mat(0, 0) = p0;
    mat(0, 1) = 0;
    mat(1, 0) = 0;
    mat(1, 1) = p1;

    /* 
       Create a general 1-qubit matrix gate wich equal to Rz(theta)
       Because 'class ClsOneQubitRotationGate' has no member named 'add_control_qubit'
    */ 
    auto gate = gate::DenseMatrix(target_qubit, mat);

    // Add controls
    for (auto c : control_qubits) {
        gate->add_control_qubit(c, 1);
    }

    circ.add_gate(gate);
}

// ------------------- Multi-controlled RY -------------------

void add_MCRY(QuantumCircuit& circ,
              const std::vector<UINT>& control_qubits,
              UINT target_qubit,
              double theta) {
    /*  RY(theta) matrix:
     [ cos(theta/2)  -sin(theta/2) ]
     [ sin(theta/2)   cos(theta/2) ] */

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    ComplexMatrix mat(2, 2);
    mat(0, 0) = c;
    mat(0, 1) = -s;
    mat(1, 0) = s;
    mat(1, 1) = c;

    //Create a general 1-qubit matrix gate which is equal to Ry(theta)
    auto gate = gate::DenseMatrix(target_qubit, mat);

    // Add controls
    for (auto cq : control_qubits) {
        gate->add_control_qubit(cq, 1);
    }

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
    circ.add_gate(X(qubit_a));

    std::vector<UINT> controls = control_qubits;
    controls.push_back(qubit_a);

    add_MCRZ(circ, controls, qubit_b, theta);


    circ.add_gate(X(qubit_a));

    add_MCRZ(circ, controls, qubit_b, -theta);
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

    circ.add_gate(H(a2));
    circ.add_gate(X(a1));
    add_Uj_dagger(circ, qx, j, lambda);

    std::vector<UINT> controls_zz;
    controls_zz.push_back(a1);
    for (UINT m = 0; m < j; ++m) {
        controls_zz.push_back(qx[m]);
    }

    const double theta_zz = (-1 * tau) / (rho_bar * l);
    add_MCRZZ(circ, controls_zz, a2, qx[j], theta_zz);

    add_Uj(circ, qx, j, lambda);
    circ.add_gate(H(a2));
    circ.add_gate(X(a1));

    add_Uj_dagger(circ, qx, j, lambda);

    std::vector<UINT> controls;
    for (UINT m = 0; m < j; ++m) {
        controls.push_back(qx[m]);
    }

    const double theta_z = (-1 * u_bar) * tau / l;
    if (!controls.empty()) {
        add_MCRZ(circ, controls, qx[j], theta_z);
    } else {
        circ.add_gate(RZ(qx[j], theta_z));
    }

    add_Uj(circ, qx, j, lambda);
}

void add_Wyj(QuantumCircuit& circ,
             UINT a1, UINT a2,
             const std::vector<UINT>& qy,
             UINT j,
             double tau,
             double rho_bar,
             double l) {
    double lambda = -M_PI / 2.0;

    circ.add_gate(X(a2));
    circ.add_gate(H(a1));
    add_Uj_dagger(circ, qy, j, lambda);

    std::vector<UINT> controls_zz;
    controls_zz.push_back(a2);
    for (UINT m = 0; m < j; ++m) {
        controls_zz.push_back(qy[m]);
    }

    const double theta_zz = (-1 * tau) / (rho_bar * l);
    add_MCRZZ(circ, controls_zz, a1, qy[j], theta_zz);

    add_Uj(circ, qy, j, lambda);
    circ.add_gate(X(a2));
    circ.add_gate(H(a1));
}



// ------------------- Wxj_tilde and Wyj_tilde blocks -------------------

void add_Wxj_tilde(QuantumCircuit& circ,
             UINT a1, UINT a2,
             const std::vector<UINT>& qx,
             UINT j,
             double tau,
             double u_bar,
             double rho_bar,
             double l) {
    double lambda = -M_PI / 2.0;
    

    //X gate on a1
    circ.add_gate(X(a1));
    
    //Uj_dagge on qx starting at j
    add_Uj_dagger(circ, qx, j, lambda);

    //multi controlled Rz on qx[j] with controlled on q[0]->q[j-1]
    std::vector<UINT> controls;
    for (UINT m = 0; m < j; ++m) {
        controls.push_back(qx[m]);
    }

    const double theta_z = (-1 * u_bar * tau) / l;
    if (!controls.empty()) {
        add_MCRZ(circ, controls, qx[j], theta_z);
    } else {
        circ.add_gate(RZ(qx[j], theta_z));
    }
    
    //Controlled Ry on a2
    std::vector<UINT> controls_ry;
    controls_ry.push_back(a1);

    const double theta_y = (-2 * tau) / (rho_bar * l);
    add_MCRY(circ, controls_ry, a2, theta_y);

    //Uj on qx starting at j
    add_Uj(circ, qx, j, lambda);

    //X gate on j
    circ.add_gate(X(qx[j]));

    //Uj_dagge on qx starting at a2
    add_Uj_dagger(circ, qx, a2, lambda);

    //multi controlled Rz on a2 with controlled on q[0]->q[j]+a1
    std::vector<UINT> controls_rz_a2;
    for (UINT m = 0; m <= j; ++m) {
        controls_rz_a2.push_back(qx[m]);
    }
    controls_rz_a2.push_back(a1);

    const double theta_z_a2 = (-2 * tau) / (rho_bar * l);
    if (!controls_rz_a2.empty()) {
        add_MCRZ(circ, controls_rz_a2, a2, theta_z_a2);
    } else {
        circ.add_gate(RZ(a2, theta_z_a2));
    }

    //Uj on qx starting at a2
    add_Uj(circ, qx, a2, lambda);

    //X gate on a1
    circ.add_gate(X(a1));

    //X gate on qz[j]
    circ.add_gate(X(qx[j]));
}

void add_Wyj_tilde(QuantumCircuit& circ,
             UINT a1, UINT a2,
             const std::vector<UINT>& qy,
             UINT j,
             double tau,
             double rho_bar,
             double l) {
    double lambda = -M_PI / 2.0;

    //X gate on a2
    circ.add_gate(X(a2));

    //X gate on qy[j]
    circ.add_gate(X(qy[j]));

    //Controlled Ry on a1
    std::vector<UINT> controls_ry;
    controls_ry.push_back(a2);

    const double theta_y = (-2 * tau) / (rho_bar * l);
    add_MCRY(circ, controls_ry, a1, theta_y);

    //Uj_dagge on qy starting at a1
    std::vector<UINT> controls_uj_dagger;
    for (UINT m = 0; m <= j; ++m) {
        controls_uj_dagger.push_back(qy[m]);
    }
    add_Uj_dagger(circ, controls_uj_dagger, a1, lambda);

    //Multi Controlled RZ on a1 with controlled on q[0]->q[j]+a2
    std::vector<UINT> controls_rz_a1;
    for (UINT m = 0; m <= j; ++m) {
        controls_rz_a1.push_back(qy[m]);
    }
    controls_rz_a1.push_back(a2);

    const double theta_z_a1 = (-2 * tau) / (rho_bar * l);
    if (!controls_rz_a1.empty()) {
        add_MCRZ(circ, controls_rz_a1, a1, theta_z_a1);
    } else {
        circ.add_gate(RZ(a1, theta_z_a1));
    }

    //Uj on qy starting at a1
    std::vector<UINT> controls_uj;
    for (UINT m = 0; m <= j; ++m) {
        controls_uj.push_back(qy[m]);
    }
    add_Uj(circ, controls_uj, a1, lambda);

    //X gate on a2
    circ.add_gate(X(a2));

    //X gate on q[j]
    circ.add_gate(X(qy[j]));
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
        add_Wxj(circ, a1, a2, qx, (UINT)j, tau, u_bar, rho_bar, l);
    }
}

void add_Qy_layer(QuantumCircuit& circ,
                  UINT a1, UINT a2,
                  const std::vector<UINT>& qy,
                  double tau,
                  double rho_bar,
                  double l) {
    UINT n = qy.size();
    for (UINT j = 0; j < n; ++j){
        add_Wyj(circ, a1, a2, qy, (UINT)j, tau, rho_bar, l);
    }
}

// ------------------- Qx_tilde(tau), Qy_tilde(tau) layers and 1 step V(tau) -------------------

void add_Qx_tilde_layer(QuantumCircuit& circ,
                  UINT a1, UINT a2,
                  const std::vector<UINT>& qx,
                  double tau,
                  double u_bar,
                  double rho_bar,
                  double l) {
    UINT n = qx.size();
    for (UINT j = 0; j < n; ++j) {
        add_Wxj_tilde(circ, a1, a2, qx, (UINT)j, tau, u_bar, rho_bar, l);
    }
}

void add_Qy_tilde_layer(QuantumCircuit& circ,
                  UINT a1, UINT a2,
                  const std::vector<UINT>& qy,
                  double tau,
                  double rho_bar,
                  double l) {
    UINT n = qy.size();
    for (UINT j = 0; j < n; ++j){
        add_Wyj_tilde(circ, a1, a2, qy, (UINT)j, tau, rho_bar, l);
    }
}

void build_lee_trotter_step(QuantumCircuit& circ,
                            int n,
                            double tau,
                            double u_bar,
                            double rho_bar,
                            double l) {
    UINT a1 = 0, a2 = 1;

    std::vector<UINT> qx(n), qy(n);
    for (int i = 0; i < n; ++i) {
        qx[i] = 2 + i;
        qy[i] = 2 + n + i;
    }

    //add_Qy_layer(circ, a1, a2, qy, tau, rho_bar, l);
    //add_Qx_layer(circ, a1, a2, qx, tau, u_bar, rho_bar, l);
    add_Qy_tilde_layer(circ, a1, a2, qy, tau, rho_bar, l);
    add_Qx_tilde_layer(circ, a1, a2, qx, tau, u_bar, rho_bar, l);

}

// ------------------- Python-visible wrapper: evolve state -------------------

py::array_t<CPPCTYPE> build_lee_initial_state(int n, double amplitude = 0.5) {
    const int n_qubits = 2 + 2 * n;
    const std::size_t dim = (std::size_t)1 << n_qubits;

    // allocate numpy array for statevector
    py::array_t<CPPCTYPE> psi_arr(dim);
    auto buf = psi_arr.request();
    auto* data = static_cast<CPPCTYPE*>(buf.ptr);

    // clear to zero
    std::fill(data, data + dim, CPPCTYPE(0.0, 0.0));

    // qubit layout
    UINT a1 = 0;
    UINT a2 = 1;
    std::vector<UINT> qx(n), qy(n);
    for (int i = 0; i < n; ++i) {
        qx[i] = 2 + i;
        qy[i] = 2 + n + i;
    }

    // grid size
    const UINT N = 1u << n;

    // 2x2 block centered on the grid:
    // start = N/2 - 1, end = start + 2
    const UINT size_block = 2;
    const UINT start = N / 2 - size_block / 2;
    const UINT end   = start + size_block;

    // classical L2 norm of the pressure field:
    // four cells each with 'amplitude'
    const double energy_norm = std::sqrt(4.0 * amplitude * amplitude);
    const double encoded_amp = (energy_norm > 0.0)
                                 ? amplitude / energy_norm
                                 : 0.0;

    // put pressure in ancilla component c = 0 (|a1 a2> = |00>)
    const UINT comp_p = 0u;

    for (UINT ix = start; ix < end; ++ix) {
        for (UINT iy = start; iy < end; ++iy) {
            std::size_t basis =
                encode_basis_index(comp_p, ix, iy, a1, a2, qx, qy);
            data[basis] = CPPCTYPE(encoded_amp, 0.0);
        }
    }

    // final safety normalization
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_sq += std::norm(data[i]);
    }
    const double norm = std::sqrt(norm_sq);
    if (norm > 0.0 && std::abs(norm - 1.0) > 1e-12) {
        for (std::size_t i = 0; i < dim; ++i) {
            data[i] /= norm;
        }
    }

    return psi_arr;
}


void build_lee_initial_state_ic(int n,
                             double amplitude,
                             QuantumState& state) {
    const int n_qubits = 2 + 2 * n;
    const std::size_t dim = (std::size_t)1 << n_qubits;

    // Access Qulacs buffer
    CPPCTYPE* data = const_cast<CPPCTYPE*>(state.data_cpp());

    // clear to zero
    std::fill(data, data + dim, CPPCTYPE(0.0, 0.0));

    // qubit layout
    UINT a1 = 0;
    UINT a2 = 1;
    std::vector<UINT> qx(n), qy(n);
    for (int i = 0; i < n; ++i) {
        qx[i] = 2 + i;
        qy[i] = 2 + n + i;
    }

    // grid size
    const UINT N = 1u << n;

    // 2x2 block centered on the grid
    const UINT size_block = 2;
    const UINT start = N / 2 - size_block / 2;
    const UINT end   = start + size_block;

    // classical L2 norm of the pressure field:
    // four cells each with 'amplitude'
    const double energy_norm = std::sqrt(4.0 * amplitude * amplitude);
    const double encoded_amp =
        (energy_norm > 0.0) ? amplitude / energy_norm : 0.0;

    // put pressure in ancilla component c = 0 (|a1 a2> = |00>)
    const UINT comp_p = 0u;

    for (UINT ix = start; ix < end; ++ix) {
        for (UINT iy = start; iy < end; ++iy) {
            std::size_t basis =
                encode_basis_index(comp_p, ix, iy, a1, a2, qx, qy);
            data[basis] = CPPCTYPE(encoded_amp, 0.0);
        }
    }

    // final safety normalization
    double norm_sq = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        norm_sq += std::norm(data[i]);
    }
    const double norm = std::sqrt(norm_sq);
    if (norm > 0.0 && std::abs(norm - 1.0) > 1e-12) {
        for (std::size_t i = 0; i < dim; ++i) {
            data[i] /= norm;
        }
    }
}



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


void evolve_lee_ic(int n,
                double tau,
                int steps,
                double u_bar,
                double rho_bar,
                double l,
                QuantumState& state) {
    const int n_qubits = 2 + 2 * n;

    // Build one Trotter step circuit
    QuantumCircuit circ(n_qubits);
    build_lee_trotter_step(circ, n, tau, u_bar, rho_bar, l);

    // Apply 'steps' times in-place
    for (int s = 0; s < steps; ++s) {
        circ.update_quantum_state(&state);
    }
}



py::array_t<CPPCTYPE> evolve_lee_default_ic(
    int n,
    double tau,
    int steps,
    double u_bar,
    double rho_bar,
    double l,
    double amplitude = 0.5) {
    const int n_qubits = 2 + 2 * n;
    const std::size_t dim = (std::size_t)1 << n_qubits;

    // Create quantum state
    QuantumState state(n_qubits);

    // Prepare initial condition in-place (Fig. 8 block)
    build_lee_initial_state_ic(n, amplitude, state);

    // Evolve in-place
    evolve_lee_ic(n, tau, steps, u_bar, rho_bar, l, state);

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

    m.def("build_lee_initial_state", &build_lee_initial_state,
          py::arg("n"),
          py::arg("amplitude") = 0.5,
          "Build the normalized initial state for the LEE test:\n"
          "p(0)=amplitude on a 2x2 block in the center, u=v=0 elsewhere.");

    m.def("evolve_lee_default_ic", &evolve_lee_default_ic,
          py::arg("n"),
          py::arg("tau"),
          py::arg("steps"),
          py::arg("u_bar"),
          py::arg("rho_bar"),
          py::arg("l"),
          py::arg("amplitude") = 0.5,
          "Build the standard Fig. 8 initial condition in C++ and evolve it\n"
          "for 'steps' Trotter steps. Returns the final statevector.");
}
