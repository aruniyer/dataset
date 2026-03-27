export FFLAGS="${FFLAGS} -fallow-argument-mismatch"
export FCFLAGS="${FCFLAGS} -fallow-argument-mismatch"
pip install 'Cython==0.29.2'
pip install 'numpy==1.13.3' 'pybind11==2.2.4'
python setup.py build_ext --inplace -j 4
python setup.py develop