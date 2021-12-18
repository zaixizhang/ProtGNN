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
