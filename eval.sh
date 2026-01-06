#!/bin/sh

########################################
########## MIDDLEBURY SECTION ##########
########################################

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
	--model_architecture patchrefiner 

########################################
########## NYU SECTION ##########
########################################

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
	--model_architecture patchrefiner

#########################################
########## HYPERSIM SECTION ##########
########################################

python -m ppd_sharpdepth.eval \
	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
	--model_architecture patchrefiner 
