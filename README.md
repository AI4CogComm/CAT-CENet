# CAT-CENet
Source codes of the article:  P. Dong *et al*., “Sensing-Aided Channel Estimation for Near-Field MIMO ISAC Systems via Cross-Attention Transformer,” *IEEE Transactions on Vehicular Technology*, vol. 75, no. 9, pp. 21796–21801, Sept. 2026.

Please cite this paper when using the codes.

# Instructions

Reading the following sections in order will help better understand all the codes.

## MATLAB

**isac_nearfield_dataset_L_3.m**  
Generates the training and test datasets for the near-field ISAC scenario with `L=3`, `K=3`, and `X=3`.

**isac_nearfield_dataset_L_6.m**  
Generates the training and test datasets for the near-field ISAC scenario with `L=6`, `K=6`, and `X=6`.

## Python

**train_isac_L=3.py**  
Trains and tests CAT-CENet for the `L=3`, `K=3`, and `X=3` scenario.

**train_isac_L=6.py**  
Trains and tests CAT-CENet for the `L=6`, `K=6`, and `X=6` scenario.

**prune_isac_L=3.py**  
Performs network pruning and evaluates the pruned CAT-CENet model.

**shap_isac_L=3.py**  
Performs SHAP-based interpretability analysis on the trained CAT-CENet model.

# Environment
These models are implemented in Keras, and the environment setting is:

-   Python 3.7.1
-   TensorFlow-gpu 2.3.0

