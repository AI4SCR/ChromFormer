# The workbench close to the library

The work bench is structures in the following way: 

`data_generation/` consists of the different data generation processes mostly the `trussart_generation_uniform.ipynb` and `uniform_fission_yeast.ipynb` that generate synthetic data used for the prediction of Trussart and Fission Yeast.

`experiments/` consists of the juptyter files used to do hyperparametrisation, and get all trussart and fission yeast results.

`models/` consists of the different data models used to infer the 3D structure mostly the `biological_trans_conf_trussart_uniform` and `biological_trans_conf_fission_yeast` that use ChromFormer to predic the 3D structure and confidences of the Trussart and Fission Yeast data.

`previous_work/` consists of the different works used to generate the necessary data that are used in experiments to evaluate ChromFormer

`saved_models/` and `saved_results/` contain the necessary saved data to reproduce ChromFormer losses and results.