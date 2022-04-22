# XAI_Fidelity_Assessment_CNN_GEO

"Gen_Synth_SHAPES.m" is the main script to generate a synthetic benchmark dataset as descirbed in Mamalakis et al (2022).
"Train&Explain_Shapes.ipynb" is the main script for building, training and explaining a convolutional network to classify every image (explanations are produced with the package "innvestigate").
"Train&Explain_Shapes_tf2.ipynb" uses tensorflow v2 to explain the already trained network with "shap".
"plotting_results_shapes.m" is the script that generates certain plots that appear in the paper. 

The above scripts are user-friendly and with many instructions and can be used to generate totally new synthetic datasets as well as train new networks from scratch.
The trained network that was used by Mamalakis et al. is provided in the file "my_model_shapes.h5". 


## Citation
Mamalakis, A., E.A. Barnes, I. Ebert-Uphoff (2022) Investigating the fidelity of explainable artificial intelligence methods for applications of convolutional neural networks in geoscience, Artificial Intelligence for the Earth Systems, also available in arXiv preprint https://arxiv.org/abs/2202.03407. 
