#! /bin/bash

for method in "genetic_algorithm" "dual_annealing"; do
    for order in $(seq 1 3); do
        echo "Runnig $method with order $order"
        echo
        python main.py --order $order --method $method
        echo
    done
done