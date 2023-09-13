#### Author: Michael Adam Shimizu, mas763@cornell.edu, 2023

# Introduction

    This folder gives instructions and documentation for the reproduction of figure 3. 

# Figure 3b

    Figure 3b shows the correlation between Fidelity and HR distance for 4-11 qubit TFIM. These figures can be reproduced by running the correlation_creator.py found in fig_3b folder. To rerun the data, simply set the rerun flag to true. To use the previous data, unzip the correlation_folder.zip and run the script. 

# Figures 3c, 3d

    These figures are heatmaps of HR distance, fidelity, and energy after VQE convergence for different one and two qubit depolarization noise values. 
    1. Figure 3c is an 11 qubit 1D-TFIM heatmap of Fidelity
    2. Figure 3d is an 11 qubit 1D-TFIM heatmap of HR distance

    To reproduce these results, in the fig_3_c_d folder, run the heatmap_creator.py script. To only reproduce the plots and not the data, unzip the hr_p1p2_plot_n_qbts_11_J_0.5_depolarization_only.zip file and run the heatmap_creator.py script. The figures should appear in the plots_p1p2_folder. 

    Rerunning the heatmap data for 11 qubit TFIM can take a long time, so to speed up, set the num_cores flag to the number of parallel cpus / parallel simulations you want to run. 

# Figure 3e
    
    Figure 3e shows the varying HR distance and fidelity as the 2-qubit gate depolarization noise is varied. To reproduce these results, just unzip the hr_p1p2_plot_n_qbts_11_J_0.5_depolarization_only.zip in the fig_3_c_d folder and run the fig_3e_creator.py script in the fig_3e folder. This figure uses the already created heatmap data, so to recheck the data you must first recreate figures 3c and 3d. See those steps above. 

