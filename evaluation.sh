#!/usr/bin/env bash
# Training script used to generate results for project

python train.py --steps 100000 --carbon_costs 237 --runs 5 --logname "final_carbon237_storage" &
python train.py --steps 100000 --carbon_costs 237 --runs 5 --no_storage --logname "final_carbon237_nostorage" &
python train.py --steps 100000 --carbon_costs 90 --runs 5 --logname "final_carbon90_storage"


for log in "final_carbon237_storage" "final_carbon237_nostorage" "final_carbon90_storage"; do
    python visuals.py --log_dir "./logs/$log"
done
