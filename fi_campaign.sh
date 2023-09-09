#!/bin/bash
#SBATCH --job-name=SCHEDULERS
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s303438@studenti.polito.it
#SBATCH --partition=global
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=/home/fpessia/Schedulers/Jetson_terminal.txt

dir=`pwd`

cd ../
source /home/fpessia/Python/bin/activate
echo "Python env setted"
cd Schedulers

for config in $dir/configs/*
do
    if [[ $config != *"default"* ]]; then
        config=$(basename ${config})
        config=${config%.*}
        mkdir -p $dir/results/$config
        rm $dir/FaultInjector/results.csv
        echo "Fault injection for $config"
        python $dir/fi_campaign.py config=$dir/configs/$config.yaml
        echo "Copying results"
        cp $dir/FaultInjector/HeatMaps/* $dir/results/$config
        cp $dir/FaultInjector/results.csv $dir/results/$config
        cp $dir/FaultInjector/ErrorModel/*  $dir/results/$config
        cp $dir/Schedulers/scheduled/*  $dir/results/$config
    fi
done