
# PD-LSTM-PCFM# Selected Experimental Data and Neural Network Architecture for Vector Hysteresis Modeling

## 1. Repository Contents

The repository includes:

- **Experimental data:** Selected magnetic flux density and magnetic field strength trajectories measured under representative magnetization conditions.
- **Neural network architecture:** Source code implementing the LSTM backbone, physics-guided characteristic feature modulation (PCFM), and Gram-based incremental reluctivity tensor construction.

## 2. Experimental Data

The experimental data describe the vector magnetic properties of electrical steel sheets under alternating and rotational magnetization.
The rolling direction (RD) and transverse direction (TD) correspond to the \(x\)- and \(y\)-axes of the constitutive model, respectively.


## 3. Neural Network Architecture

The proposed model is a history-dependent inverse vector hysteresis model that predicts the incremental reluctivity tensor rather than directly predicting magnetic field strength.
