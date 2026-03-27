pip install numpy==1.15.4 scipy==1.1.0 cython==0.25.2
sed -i 's|Configuration("metrics/cluster"|Configuration("metrics.cluster"|' \
    sklearn/metrics/cluster/setup.py
python setup.py build_ext --inplace
pip install -e . --no-build-isolation