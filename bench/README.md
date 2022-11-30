# The workbench close to the library

The work bench is structured in the following way: 

```demo``` contains an end to end simple example of Trussart data generation and training of model with evaluation and calibration

`data_generation/` consists of the different data generation processes mostly the `trussart_generation_uniform.ipynb` and `uniform_fission_yeast.ipynb` that generate synthetic data used for the prediction of Trussart and Fission Yeast.

`models/` consists of the different data models used to infer the 3D structure mostly the `biological_trans_conf_trussart_uniform` and `biological_trans_conf_fission_yeast` that use ChromFormer to predic the 3D structure and confidences of the Trussart and Fission Yeast data.

`experiments/` consists of the juptyter files used to do hyperparametrisation, and get all trussart and fission yeast results.

`previous_work/` consists of the different works used to generate the necessary data that are used in experiments to evaluate ChromFormer

`saved_models/` and `saved_results/` contain the necessary saved data to reproduce ChromFormer losses and results.

In order to use notebooks, a .env file at the root of your work must be created with the variable DATA_DIR="path to the data folder"