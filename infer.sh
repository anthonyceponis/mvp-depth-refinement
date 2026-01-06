########################################
########## MIDDLEBURY SECTION ##########
########################################

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture unidepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture zoedepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "andrew-healey/sharpdepth" \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture pixelperfectdepth_unidepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "andrew-healey/sharpdepth" \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture pixelperfectdepth_zoedepth


#python -m ppd_sharpdepth.infer \
#	--checkpoint "OHo315/PatchRefiner" \
#	--dataset_config_path config/dataset_depth/data_middlebury_test.yaml \
#	--model_architecture patchrefiner


########################################
########## NYU SECTION ##########
########################################

#python -m ppd_sharpdepth.infer \
#	--checkpoint submodules/SharpDepth/checkpoints/sharpdepth \
#	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
#	--model_architecture sharpdepth_ppd_unidepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
#	--model_architecture unidepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
#	--model_architecture zoedepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "andrew-healey/sharpdepth" \
#	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
#	--model_architecture pixelperfectdepth

# python -m ppd_sharpdepth.infer \
# 	--checkpoint "OHo315/PatchRefiner" \
# 	--dataset_config_path config/dataset_depth/data_nyu_test.yaml \
# 	--model_architecture patchrefiner

#########################################
########## HYPERSIM SECTION ##########
########################################

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
#	--model_architecture unidepth

#python -m ppd_sharpdepth.infer \
#	--checkpoint "lpiccinelli/unidepth-v1-vitl14" \
#	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
#	--model_architecture zoedepth

python -m ppd_sharpdepth.infer \
	--checkpoint "andrew-healey/sharpdepth" \
	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
	--model_architecture pixelperfectdepth \
	--subset_size 5

#python -m ppd_sharpdepth.infer \
#	--checkpoint "OHo315/PatchRefiner" \
#	--dataset_config_path config/dataset_depth/data_hypersim_test.yaml \
#	--model_architecture patchrefiner
