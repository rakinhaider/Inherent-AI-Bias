#!/bin/bash

printf "Table 1: Model performances for equally
separable unprivileged group with 2 attributes\n"

python -m without_resource_constraints --sigma-1 2 --sigma-2 2

printf "\nTable 2: Model performances for equally
separable unprivileged group with 10 attributes\n"

python -m without_resource_constraints --sigma-1 4 --sigma-2 4\
 	--n-redline 3 --n-feature 4

printf "\nTable 3: Model performances for less
separable unprivileged group with 2 attributes\n"
printf "\nNB\n"
python -m without_resource_constraints
printf "\nSVM\n"
python -m without_resource_constraints --estimator svm
printf "\nDT_5\n"
python -m without_resource_constraints --estimator dt
printf "\nPR\n"
python -m without_resource_constraints --estimator pr
printf "\nRed\n"
python -m without_resource_constraints --reduce

printf "\nTable 4: Model performances for less
separable unprivileged group with 10 attributes\n"
python -m without_resource_constraints --sigma-1 4 --sigma-2 7\
 	--n-redline 3 --n-feature 4

printf "\nTable 5: Model performances for less separable
unprivileged group with 2 features and insufficient resources\n"
printf "\nNB\n"
python -m with_resource_constraints -r Low
printf "\nSVM\n"
python -m with_resource_constraints -r Low --estimator svm
printf "\nDT_5\n"
python -m with_resource_constraints -r Low --estimator dt
printf "\nPR\n"
python -m with_resource_constraints -r Low --estimator pr
printf "\nRed\n"
python -m with_resource_constraints -r Low --reduce

printf "\nTable 6: Model performances for less separable
unprivileged group with 2 features and surplus resources\n"
printf "\nNB\n"
python -m with_resource_constraints -r High
printf "\nSVM\n"
python -m with_resource_constraints -r High --estimator svm
printf "\nDT_5\n"
python -m with_resource_constraints -r High --estimator dt
printf "\nPR\n"
python -m with_resource_constraints -r High --estimator pr
printf "\nRed\n"
python -m with_resource_constraints -r High --reduce

printf "\nTable 7: Model performances on COMPAS original
and de-biased\n"
python inherent_bias/exp7/fair_ml.py

printf "\nTable 8: Model performances for COMPAS SFBD
de-biased by inflating privileged unfavored class\n"
python -m oversampled_compas_experiment --sample-mode 1

printf "\nTable 9: Model performances for COMPAS SFBD
de-biased by inflating unprivileged favored class\n"
python -m oversampled_compas_experiment --sample-mode 2
