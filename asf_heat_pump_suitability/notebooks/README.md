# asf_heat_pump_suitability/notebooks

## reweight_epc_test.py

To make statements about heat pump suitability by area, the sample of properties we have in our baseline EPC dataset must be representative
of all properties in that area. However, our assumption is that this is not the case for the EPC samples we have.
This is due to sample size for some areas, and potentially inherent bias in the way EPC is generated because properties
only receive an EPC certificate under certain conditions (renting/selling).

In `reweight_epc_test.py`, we test the Iterative Proportional Fitting (IPF) algorithm to reweight EPC data from some sample LSOAs
taken from the EPC dataset. We use target marginals from 2021 census data for two features: `tenure` and `property_type`.

Packages used for IPF: Python package [balance](https://github.com/facebookresearch/balance) by Facebook Research which repackages [ipfn](https://github.com/Dirguis/ipfn).

### Setup

Run the following:

```
make install
python -m ipykernel install --user --name=asf_heat_pump_suitability
jupytext --to ipynb asf_heat_pump_suitability/notebooks/reweight_epc_test.py
```
