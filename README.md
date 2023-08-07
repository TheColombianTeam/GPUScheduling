# Schedulers

This framework is a functional simulator of different scheduler policies. The scheduler policies implemented are based on[ 1](http://arxiv.org/abs/1808.07984).

## Intallation

Please clone this repo using the below command:

```bash
git clone --recurse-submodules https://github.com/TheColombianTeam/Schedulers.git
```

This framework is built on top of [PyOpenTCU](https://github.com/TheColombianTeam/PyOpenTCU.git). The requirements are available inside [requeriments.txt](./requeriments.txt).

If you are using Anaconda or Miniconda as the environment manager, please install the requirements as follows:

```bash
conda env create -f environment.yml
```

Otherwise, use the following command:

```bash
pip install -r requeriments.txt
```

Those commands should create the conda environment with whole packages required to excuse this framework. However, if exist any problem with the SFPY library, please read the documentation available on [2](https://github.com/billzorn/sfpy.git).

## Usage

Once the packages are installed, run the following command:

```bash
sh run.sh
```

This bash script runs the validation process. This process executes the default *Mock* scheduler and compares the result with the golden result.

To validate another scheduler, please modify the `SCHEDULER_POLICY` variable inside the `run.sh` file. The possible options available are:

```bash
SCHEDULER_POLICY=DistributedBlock
SCHEDULER_POLICY=DistributedCTA
SCHEDULER_POLICY=GlobalRoundRobin
SCHEDULER_POLICY=Greedy
SCHEDULER_POLICY=TwoLevelRoundRobin
```

### **NOTE: These policies are under development**

### Development process

Inside the [Schedulers](/Schedulers/) module, you could find the different policies to be implemented. The [Mock](/Schedulers/mock.py)  class contains the basic idea to be developed. Inside this class, you will find the [scheduler_algorithm](/Schedulers/mock.py?plain=1#L19) method. This method must be implemented on the other schedulers. This method has as parameters the `A`, `B`, and `C` matrices and should return an array with the CTAs structure shown in [Scheduler](/Schedulers/models/Scheduler.py?plain=1#L20). This structure must be returned by all the policies implemented since it is used by the Fault Injector (under development).

To validate the scheduler policy implemented, please modify the `SCHEDULER_POLICY` variable on `run.sh` according to the information above.

