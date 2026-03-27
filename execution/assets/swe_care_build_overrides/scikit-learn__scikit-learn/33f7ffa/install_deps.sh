pip install numpy==1.15.4 scipy==1.1.0 cython==0.29.37
CLUSTER_SETUP="sklearn/metrics/cluster/setup.py"
sed -i 's|Configuration("metrics/cluster"|Configuration("metrics.cluster"|' \
    "$CLUSTER_SETUP"
python setup.py build_ext --inplace
pip install -e . --no-build-isolation