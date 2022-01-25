#!/bin/bash

printf "Table 1: Model performances for equally
separable unprivileged group with 2 attributes\n"

python3 -m without_resource_constraints --sigma-1 2 --sigma-2 2

printf "\nTable 2: Model performances for equally
separable unprivileged group with 10 attributes\n"

python3 -m without_resource_constraints --sigma-1 4 --sigma-2 4\
 	--n-redline 3 --n-feature 4

printf "\nTable 3: Model performances for less
separable unprivileged group with 2 attributes\n"

python3 -m without_resource_constraints

printf "\nTable 4: Model performances for less
separable unprivileged group with 10 attributes\n"
python3 -m without_resource_constraints --sigma-1 4 --sigma-2 7\
 	--n-redline 3 --n-feature 4

printf "\nTable 5: Model performances for less separable
unprivileged group with 2 features and insufficient resources\n"
python3 -m with_resource_constraints -r Low

printf "\nTable 6: Model performances for less separable
unprivileged group with 2 features and surplus resources\n"
python3 -m with_resource_constraints -r High

printf "\nTable 7: Model performances on COMPAS original
and de-biased\n"
python3 inherent_bias/exp7/fair_ml.py

printf "\nTable 8: Model performances for COMPAS SFBD
de-biased by inflating privileged unfavored class\n"
python3 -m oversampled_compas_experiment --sample-mode 1

printf "\nTable 9: Model performances for COMPAS SFBD
de-biased by inflating unprivileged favored class\n"
python3 -m oversampled_compas_experiment --sample-mode 2
