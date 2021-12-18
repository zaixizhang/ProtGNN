#!/bin/bash
for clst in 0.0 0.01 0.05 0.10 0.50

do
	for sep in 0.0 0.01 0.05 0.10 0.50
	do
		echo $clst
		echo $sep
		python -m models.train_gnns --clst ${clst} --sep ${sep}
	done
	
done
#
#python -m models.train_gnns train_GC.train_MUTAG()  \
#        --dataset_name mutag --model_name gin \
#        --c_puct 10.0 --reward_method mc_l_shapley
#
#python -m models.train_gnns train_GC \
#        --dataset_name BA_2Motifs --random_split True \
#        --latent_dim 20 20 20 --adj_normlize False \
#        --emb_normlize True --readout mean --c_puct 10.0 \
#        --min_atom 5 --reward_method mc_l_shapley
#
#python -m models.train_gnns train_GC
#        --dataset_name Graph-SST2 --model_name gat \
#        --random_split False --reward_method mc_l_shapley
#
#python -m models.train_gnns train_NC \
#        --dataset_name BA_shapes --random_split True \
#        --latent_dim 20 20 20 --concate True --adj_normlize False --emb_normlize True \
#        --high2low True --min_atom 5 --reward_method nc_mc_l_shapley
