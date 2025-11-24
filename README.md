# Implementation of Algorithm from arXiv:2510.17978

This repository provides an implementation of the algorithm described in **arXiv:2510.17978**, using both **Qulacs** and **Qiskit**. The script `PDE_for_cfd.py` reproduces *Figure 8* from the paper.

---

## Features

* Quantum simulation tools implemented in **Qulacs** and **Qiskit**
* Reproduction of Figure 8 from the referenced paper
* Adjustable grid size and simulation parameters

---

## File Overview

* **`PDE_for_cfd.py`** — Main script implementing and plotting the result corresponding to *Figure 8*.
* **`cpp_lib_lee_trotterization/`** — C++ implementation for speeding up the simulation, built as a library.

---

## Installation & Build Instructions

### 1. Install Qulacs

Clone the Qulacs repository:

```bash
git clone https://github.com/qulacs/qulacs.git
```

Ensure you update the Qulacs path inside your `CMakeLists.txt`.

### 2. Build the C++ Library

From the `cpp_lib_lee_trotterization` directory:

```bash
mkdir build/
cd build/
cmake ../
make
cmake --install .
```

Move or reference the resulting library wherever your Python script expects it.

---

## Running the Simulation

To generate Figure 8:

```bash
python3 PDE_for_cfd.py
```

---

## Configuration

You can modify:

* Grid size
* Physical/algorithmic parameters

These settings are located at the top of the `PDE_for_cfd.py` script.

---

## Reference

This work implements methods described in:

* *arXiv:2510.17978*

---

## CopyRight 

© 2025 Alexandre Gallardo – All rights reserved

---





