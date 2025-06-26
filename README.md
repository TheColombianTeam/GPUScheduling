# PyOpenGPU_Scheduling Framework

This framework provides a functional structural model of modern GPUs, with a primary focus on executing workloads on Tensor Cores, such as matrix multiplication operations. It enables users to simulate and analyze the GPU execution behavior with a particular emphasis on scheduling policies for Tensor Core workloads.

Key features include:

Support for multiple state-of-the-art scheduling policies, inspired by real-world hardware implementations in modern GPUs [1](https://ieeexplore.ieee.org/abstract/document/8625517).

Configurable architectural components to explore design trade-offs and performance implications.

This framework is ideal for researchers and developers interested in studying GPU microarchitecture, especially in the context of AI and HPC workloads.


# Schedulers

The workload scheduling implement the tiling approach presented in [Tiling algorithm GEMM](http://arxiv.org/abs/1808.07984) and [NVIDIA Tiling](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), 

# Functional Cores (Tensor Cores)

This framework uses the Tensor Core descriptions and models by [PyOpenTCU](https://github.com/TheColombianTeam/PyOpenTCU.git).

## Install

Please clone this repo using the following command:

```bash
git clone --recurse-submodules https://github.com/TheColombianTeam/Schedulers.git
```

This framework is built on top of [PyOpenTCU](https://github.com/TheColombianTeam/PyOpenTCU.git), and the requirements are available inside [requeriments.txt](./requeriments.txt).

Please install using following command:

```bash
pip install -r requeriments.txt
```

Those commands should install the dependencies required to excuse this framework. However, if exist any problem with the *SFPY* library, please read the documentation available on [sfpy](https://github.com/billzorn/sfpy.git).

## Usage

Once the packages are installed, run the following command:

```bash
sh fi_campaign.sh
```

This bash script runs the Fault Injection campaings. This process executes all the configurations under the *Mock* scheduler and compares the result with the golden result.

To validate another scheduler, please use the `scheduler` argument when execute when you run the code. The possible options available are:

```bash
SCHEDULER_POLICY=DistributedBlock
SCHEDULER_POLICY=DistributedCTA
SCHEDULER_POLICY=GlobalRoundRobin
SCHEDULER_POLICY=Greedy
SCHEDULER_POLICY=TwoLevelRoundRobin
```

Moreover, in the [configs](./configs/) folder you can find several GPU configurations.

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
