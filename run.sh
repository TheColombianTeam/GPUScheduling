SCHEDULER_POLICY=TwoLevelRoundRobin

python validate.py scheduler=$SCHEDULER_POLICY

SCHEDULER_POLICY=GlobalRoundRobin

python validate.py scheduler=$SCHEDULER_POLICY