# Double Pendulum Simulator

A simulation project modeling the dynamics of a double pendulum system.

## Description

![image](https://github.com/user-attachments/assets/9a9404a8-33ee-43cb-96b2-19e8e1619b91)

**README IN PROGRESS**

This project simulates the motion of a double pendulum — a system known for its rich dynamic behavior and sensitivity to initial conditions.
Since the system cannot be solved analitically, numerically is the way. The pendulum moves in two axis - horizontal and vertical.
The necessary initial conditions are in form of a *record* consisting of four real numbers being (θ<sub>1</sub>, θ<sub>2</sub>, ω<sub>1</sub>, ω<sub>2</sub>) - angles and angular velocities of both pendula.
Arbitrary number of records can be generated and fed to the program to be processed in parallel. Those are grouped in the initial basket. 
A basket stores records of all pendula for the specific moment in time.
Program can operate in two modes - _storage_ and _visual_.

Storage mode allows generating raw data. Every new iteration generates a new record which is then further used to generate the following and so on - meaning every record is dependent on the previous one.
Data generated on the GPU is stored in a memory mapped file.

Visual mode allows visualising pendulum motion. Data is being processed and represented on the screen in parallel thanks to double buffers. (comming soon)

## Prerequisites

Ensure you have the following installed:
- C++ Compiler (C++20)
- Make
- Python 3.x
- GLEW, GLFW, GLUT
- CUDA SDK
- nVidia drivers (make sure Cuda version of nvcc and driver match)

Make sure OpenGL is set to use nVidia GPU and not the integrated one.
Make sure everything above is in the PATH and LD_LIBRARY_PATH.
  
## Build

Simpely run ```make``` after all the prerequisits are satisfied. By default all the code is compiled for fp32. To switch to fp64 add ```REAL_TYPE=1``` in make command.

There are targets: gen, int, visual, storage, for generating just the corresponding parts of the code.

**Change the path to cuda_runtime.h in Makefile (-L flag) (Working on it).**
**VISUAL CANT COMPILE FOR FP64**

## Features

- Random Record Generation\
  ```bin/rand_generator.exe <input_file> <record_count> [<options>]```\
  This executable can generate random input data. The range of records can be further narrowed down via options e.g. ```-teta1 -0.2 0.4``` narrows down the range of randomly generated θ<sub>1</sub> values to [-0.2, 0.4]. Output data is by default stored in output.bin, but this can be configured with ```-o <output_file>``` option.
- Binary Representation\
  ```bin/bin_interpret.exe <input_file> <record_count> [<options>]```\
  This executable outputs the ```record_count``` records of finary file ```input_file``` as human-readable strings. Option ```-s <step_size> <choice>``` allows for every ```step_size``` record starting from ```choice``` to be outputed. This is useful for extracting the data of single pendulum instead of whole baskets.
- Storage mode\
  ```bin/storage.exe <input_file> <record_count> [<options>]```\
  This executable generates approximation data on the GPU, transfers it to the CPU in page-locked memory and eventually passes it to the memory mapped file (can be specialized via ```-o <output_file>```). There is a list of options for configuring simulation constants: lenghts (l<sub>1</sub>, l<sub>2</sub>) and masses (m<sub>1</sub>, m<sub>2</sub>) of pendula as well as gravitational pull (g) and approximation frequency (h). Instance: ```-m1 0.9```. 
- Visual mode\
  ```bin/storage.exe <input_file> <record_count> [<options>]```\
  This executable visualises the simulation through OpenGL. The command line arguments are as for storage mode.

