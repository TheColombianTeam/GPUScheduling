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
        cp $dir/Schedulers/DynamicBehaviour/*.png  $dir/Schedulers/DynamicBehaviour/$config
        cp $dir/Schedulers/DynamicBehaviour/*.csv  $dir/Schedulers/DynamicBehaviour/$config
    fi
done

rm $dir/Schedulers/DynamicBehaviour/*.png
rm $dir/Schedulers/DynamicBehaviour/*.csv