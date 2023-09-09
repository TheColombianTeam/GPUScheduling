#!/bin/bash
#SBATCH --job-name=SCHEDULERS
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s303438@studenti.polito.it
#SBATCH --partition=global
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=/home/fpessia/Schedulers/ErrorInjector/Modellor_terminal.txt


cd ../
source /home/fpessia/Python/bin/activate
echo "Python env setted"
cd Schedulers


dir=`pwd`

for config in $dir/configs/*
do
    if [[ $config != *"default"* ]]; then
        config=$(basename ${config})
        config=${config%.*}
        mkdir -p $dir/ErrorInjector/$config
        echo "Modelling faults for $config"
        python $dir/GeneratingModels.py config=$dir/configs/$config.yaml
        echo "Copying results"
        mv $dir/ErrorInjector/Models/*  $dir/ErrorInjector/$config

    fi
done