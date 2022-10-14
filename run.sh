#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL=3

printf "Table I: MODEL PERFORMANCES ON EQUALLY SEPARABLE UNPRIVILEGED GROUP\n"

python3 -m without_resource_constraints --sigma-1 2 --sigma-2 2

printf "\nTable II: MODEL PERFORMANCES ON LESS SEPARABLE UNPRIVILEGED GROUP\n"
printf "\nMethod: NB\n"
python3 -m without_resource_constraints
printf "\nMethod: SVM\n"
python3 -m without_resource_constraints --estimator svm
printf "\nMethod: DT_5\n"
python3 -m without_resource_constraints --estimator dt
printf "\nMethod: PR\n"
python3 -m without_resource_constraints --estimator pr
printf "\nMethod: RBC\n"
python3 -m without_resource_constraints --reduce

printf "\nTable III: MODEL PERFORMANCES ON LESS SEPARABLE UNPRIVILEGED GROUP 
WITH SCARCE OR SURPLUS RESOURCES\n"
printf "\nResource: Scarce\n"
python3 -m with_resource_constraints -r Low

printf "\nResource: Surplus\n"
python3 -m with_resource_constraints -r High

printf "\nTable IV: MODEL PERFORMANCES FOR COMPAS SFBD DE-BIASED BY INFLATING
PRIVILEGED OR UNPRIVILEGED UNFAVORED CLASS\n"
printf "\nInflated Class: Privileged\n"
python3 -m oversampled_compas_experiment --sample-mode 1
printf "\nInflated Class: Unprivileged\n"
python3 -m oversampled_compas_experiment --sample-mode 2

mkdir -p outputs/ppds/
printf "\nTable V: PREDICTIVE POWERS OF EACH FEATURE IN COMPAS DATASET\n"
python3 -m predictive_power_difference --dataset compas >/dev/null
awk -F'\t' 'BEGIN{OFS="\t";}{print $1, $2, $5, $8;}' outputs/ppds/compas.csv
printf "\n"

printf "#################### Additional results ############################"

printf "\nTable a: MODEL PERFORMANCES ON EQUALLY 
SEPARABLE UNPRIVILEGED GROUP WITH 10 ATTRIBUTES\n"

python3 -m without_resource_constraints --sigma-1 4 --sigma-2 4\
 	--n-redline 3 --n-feature 4

printf "\nTable b: Model performances for less
separable unprivileged group with 10 attributes\n"
python3 -m without_resource_constraints --sigma-1 4 --sigma-2 7\
 	--n-redline 3 --n-feature 4

printf "\nTable c: Model performances for less separable 
unprivileged group with 2 features with insufficient resources.\n"
printf "\nMethod: SVM\n"
python3 -m with_resource_constraints -r Low --estimator svm
printf "\nMethod: DT_5\n"
python3 -m with_resource_constraints -r Low --estimator dt
printf "\nMethod: PR\n"
python3 -m with_resource_constraints -r Low --estimator pr
printf "\nMethod: RBC\n"
python3 -m with_resource_constraints -r Low --reduce

printf "\nTable d: Model performances for less separable
unprivileged group with 2 features and surplus resources\n"
printf "\nMethod: SVM\n"
python3 -m with_resource_constraints -r High --estimator svm
printf "\nMethod: DT_5\n"
python3 -m with_resource_constraints -r High --estimator dt
printf "\nMethod: PR\n"
python3 -m with_resource_constraints -r High --estimator pr
printf "\nMethod: RBC\n"
python3 -m with_resource_constraints -r High --reduce

printf "\nTable e: Model performances on COMPAS original
and de-biased\n"
python3 inherent_bias/exp7/fair_ml.py

printf "\nTable f: Predictive powers of features in German Credit and 
Bank dataset\n"
printf "\nDataset: Bank\n"
python3 -m predictive_power_difference --dataset bank >/dev/null
awk -F'\t' 'BEGIN{OFS="\t";}{print $1, $2, $5, $8;}' outputs/ppds/compas.csv
printf "\nDataset: German Credit\n"
python3 -m predictive_power_difference --dataset german >/dev/null
awk -F'\t' 'BEGIN{OFS="\t";}{print $1, $2, $5, $8;}' outputs/ppds/compas.csv

