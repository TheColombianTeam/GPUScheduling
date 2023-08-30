dir=`pwd`

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
    fi
done
