# Schedulers

This framework is a functional simulator of different scheduler policies for Matrix Multiplication. The scheduler policies implemented the tiling approach presented in [Tiling algorithm GEMM](http://arxiv.org/abs/1808.07984) and [NVIDIA Tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), and it describes the functional implementation of different scheduling policies presented by [Scheduling for NoC arhcitectures](https://ieeexplore.ieee.org/abstract/document/8625517)

## Intallation

Please clone this repo using the following command:

```bash
git clone --recurse-submodules https://github.com/TheColombianTeam/Schedulers.git
```

This framework is built on top of [PyOpenTCU](https://github.com/TheColombianTeam/PyOpenTCU.git). The requirements are available inside [requeriments.txt](./requeriments.txt).

Please install using following command:

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

Inside the [Schedulers](/Schedulers/) module, you can find the different policies to be implemented. The [Mock](/Schedulers/mock.py) class contains the basic idea of: 
- Basic Tiling (generates CTAs Id and (X, Y) coordinates, other parameters are dummy)
- return a list of dictionaries with the scheduled tiles (No scheduler implemented) 

Inside the *Mock* class, you will find the method [scheduler_algorithm](/Schedulers/mock.py?plain=1#L19). This method is the one that should implement the tiling procedure and the scheduling implementation, and it should return the list of dictionaries as the requirement for the fault injector. This method is also the interface between the scheduler and the rest of the framework, so you should use a similar implementation of such a function for every scheduler you need to implement. This method receives as parameters the matrices `A`, `B`, and `C` and should return an array with the CTAs structure shown in [Scheduler](/Schedulers/models/Scheduler.py?plain=1#L20). All the policies implemented must return this structure since it is used by the Fault Injector (now under development).

To validate the scheduler policy implemented, please modify the `SCHEDULER_POLICY` variable on `run.sh` according to the information above. After the execution, you can find in the ./logs directory a log file that will show if the verification passed or, on the contrary, it failed. 

NOTE: the verification only checks that the matrix multiplication can be done by using the array structure your scheduler implementation provided; however, the correct scheduling assignment must be checked on your own. 

### Change Matrix Size
If you want to evaluate different matrix sizes, you can modify the arguments of the matrix generation in the [golden.py](https://github.com/TheColombianTeam/Schedulers/blob/c50d9a0069a373c0de62c5f91d4092b75af9afcb/golden.py#L43C12-L43C12). For example, the following snippet code for the main function creates the golden matrices a,b,c,d of size 128X128, you can use any size you want being careful that the sizes are correct for performing matrix multiplication.

```python
def main():
    a = create_matrix(size=[128,128])
    b = create_matrix(size=[128,128])
    c = create_matrix(size=[128,128])
    d_ = tiling(a, b, c)
    d = np.matmul(a, b) + c
    validate(d, d_)
    save_matrix(a, "a")
    save_matrix(b, "b")
    save_matrix(c, "c")
    save_matrix(d_, "d")
```
You can generate the new matrices once you modify the golden file by executing the following command. NOTE: If you increase the matrix size, this golden generation will take from some minutes to several hours since the whole multiplication will run on the tensor unit. 

```bash
python golden.py
```
After the golden is generated successfully, then you can run the bash script to check if the array of CTAs generated in your scheduler policy is still working correctly. 

### Formatting code

Only for maintaining a common structure, I recommend using one format extension (i.e., [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)) for vs. Please use it.

### **MODIFICATIONS MY LAST PUSH

- Implemented golden module: uses function in Kernel repo to calculate the golden value matrix mul and compare against matrix multiplication performed with np.matmul.  Furthermore, it calls the schedulers to allocate CTAs to SMs and validates scheduler behaviour --> scheduler results stored in .json as suggested

```python
def golden(A_x_dim, A_B_common_dim, B_y_dim, min_value = 0, max_value = 1):
```

the inptus of this module main function are the input matrix dimentions and the minimum and maximum value of the normalized distribution that are 
going to effect scale and offset in function "create matrix"

- fault list: randomly generate (using the numeric sequence suggested by SFI paper) 8k faults. If in the numeric sequence there are more than 380 similar faults the sequence is generated again otherwise the faults that are occuring more then once are replace deterministically in order to make sure to inject different faults. The same faults are injected on different schedulers to observe how the same failure generates errors at output tensor --> time of execution of this module depends on the required time to generate randly a numeric seed that will produce sequence of fault IDs to inject with less than 380 repetitions

```python
def fault_list(faulty_SM = 0, faulty_cluster = 0):
```
through this function user can set the faulty HW, please no double checks are done on weather or not the specified hw exists--> if not during fault 
injection nothing is going to be injected

-Injector: this function reads input matrix, json files generated by schedulers and fault list. Through a number of parallel process set by user though input parameter of this module, faults are injected and results are stored in .csv file

 
```python
def injector(number_of_workers):
```

number_of_workers = number of parallel process that are working symultaneously -->user might decide its more siutable value thorugh some injections and according to fastest exe sets this parameter

-validator: this module needs output tensor dimentions as input in order to generate a heat map with same dimentions, some heat maps have been generated using data from 10 fault injections in order to check code validity, please deleate HeatMap repo before starting a serius fault injection
### **Future modifications
