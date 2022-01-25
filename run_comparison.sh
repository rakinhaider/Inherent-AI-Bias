#!/bin/bash

printf "\nTable x: Model performances for Less Separable privileged group\n"

python3 -m without_resource_constraints
# python3 -m without_resource_constraints --estimator svm
# python3 -m without_resource_constraints --estimator dt
# python3 -m without_resource_constraints --estimator pr
# python3 -m without_resource_constraints --estimator nb --reduce

printf "\nTable y: Model performances for Less Separable
privileged group with Low Resource\n"

# python3 -m with_resource_constraints
# python3 -m with_resource_constraints --estimator svm
# python3 -m with_resource_constraints --estimator dt
# python3 -m with_resource_constraints --estimator pr
# python3 -m with_resource_constraints --estimator nb --reduce

printf "\nTable z: Model performances for Less Separable
privileged group with high resources\n"

# python3 -m with_resource_constraints
# python3 -m with_resource_constraints --estimator svm
# python3 -m with_resource_constraints --estimator dt
# python3 -m with_resource_constraints --estimator pr
# python3 -m with_resource_constraints --estimator nb --reduce
