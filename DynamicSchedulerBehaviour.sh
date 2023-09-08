dir=`pwd`

for config in $dir/configs/*
do
    if [[ $config != *"default"* ]]; then
        config=$(basename ${config})
        config=${config%.*}
        mkdir -p $dir/Schedulers/DynamicBehaviour/$config
        echo "Schedulers dynamic behaviour for $config"
        python $dir/SchedulerDynamicBehaviour.py config=$dir/configs/$config.yaml
        echo "Copying results"
        cp $dir/Schedulers/DynamicBehaviour/DistributedBlock.png  $dir/Schedulers/DynamicBehaviour/$config
        cp $dir/Schedulers/DynamicBehaviour/DistributedCTA.png  $dir/Schedulers/DynamicBehaviour/$config
        cp $dir/Schedulers/DynamicBehaviour/Greedy.png  $dir/Schedulers/DynamicBehaviour/$config
        cp $dir/Schedulers/DynamicBehaviour/TwoLevelRoundRobin.png  $dir/Schedulers/DynamicBehaviour/$config
        cp $dir/Schedulers/DynamicBehaviour/GlobalRoundRobin.png  $dir/Schedulers/DynamicBehaviour/$config
    fi
done