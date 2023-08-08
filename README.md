# Schedulers

This framework is a functional simulator of different scheduler policies for Matrix Multiplication. The scheduler policies implemented the tiling approach presented in [Tiling algorithm GEMM](http://arxiv.org/abs/1808.07984), and it describes the functional implementation of different scheduling policies presented by [Scheduling for NoC arhcitectures](https://ieeexplore.ieee.org/abstract/document/8625517)

## Intallation

Please clone this repo using the following command:

```bash
git clone --recurse-submodules https://github.com/TheColombianTeam/Schedulers.git
```

This framework is built on top of [PyOpenTCU](https://github.com/TheColombianTeam/PyOpenTCU.git). The requirements are available inside [requeriments.txt](./requeriments.txt).

If you are using Anaconda or Miniconda as the environment manager, please install the requirements as any of the following commands:

```bash
conda env create -f environment.yml
```

Otherwise, use the following command:

```bash
pip install -r requeriments.txt
```

Those commands should create the conda environment with whole packages required to excuse this framework. However, if exist any problem with the SFPY library, please read the documentation available on [sfpy](https://github.com/billzorn/sfpy.git).

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

Inside the [Schedulers](/Schedulers/) module, you can find the different policies to be implemented. The [Mock](/Schedulers/mock.py) class contains the basic idea of: i) tiling and ii) a list of dictionaries with the scheduled tiles to be returned. Inside the *Mock* class, you will find the method [scheduler_algorithm](/Schedulers/mock.py?plain=1#L19). This method is the one that should implement the tiling procedure and the scheduling implementation, and it should return the list of dictionaries as the requirement for the fault injector. This method is also the interface between the scheduler and the rest of the framework, so you should use a similar implementation of such a function for every scheduler you need to implement. This method receives as parameters the matrices `A`, `B`, and `C` and should return an array with the CTAs structure shown in [Scheduler](/Schedulers/models/Scheduler.py?plain=1#L20). All the policies implemented must return this structure since it is used by the Fault Injector (now under development).

To validate the scheduler policy implemented, please modify the `SCHEDULER_POLICY` variable on `run.sh` according to the information above. After the execution, you can find in the ./logs directory a log file that will show if the verification passed or, on the contrary, it failed. NOTE: the verification only checks that the matrix multiplication can be done by using the array structure your scheduler implementation provided; however, the correct scheduling assignment must be checked on your own. 

