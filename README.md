Implementation of the algorithm explain in the arXiv:2510.17978 in Qulacs and
Qiskit.


PDE_for_cfd.py implement the fig 8. 

How to use it :
first install the qulacs git : git clone https://github.com/qulacs/qulacs.git
Change the good directory in the CMakeLists.txt

then build and move the lib where you want to use it:
    cd cpp_lib_lee_trotterization/
    mkdir build/
    cd build/
    cmake ../
    make
    cmake --install .

"python3 PDE_for_cfd.py" to show the fig 8.

you can change the size of the grid and the parameters in the top of the file PDE_for_cfd.py


