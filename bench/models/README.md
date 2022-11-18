# In this folder you will find examples on how to train on the synthetic data to predict structures from experimental data. 

`ablation_chromformer_with_tech_data` and `ablation_tech_with_Cluster_Walk` are folders that contain the ablation study which we used simply to find how much each of our ideas improved from the previous model called TECH-3D

`biological_linear_trussart`, `biological_linear_trussart_special_values`, `biological_linear_uniform_confidence_trussart_linear` and  `uniform_linear_trussart` are folders containing tests we performed as we were improving the model and creating new features. They do not contain ChromFormer and are not the the model used in the paper.

`biological_trans_conf_trussart_uniform` and  `biological_trans_conf_fission_yeast` are the folders that contain ChromFormer trained on ClusterWalk synthetically generated data to predict the trussart structure and the fission yeast structure. They each contain confidence prediction as well as calibration models.