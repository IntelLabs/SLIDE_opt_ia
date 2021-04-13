# SLIDE

The SLIDE package contains the source code for reproducing the main experiments in this [paper](https://arxiv.org/abs/1903.03129).

## Dataset

The Datasets can be downloaded in [Amazon-670K](https://drive.google.com/open?id=0B3lPMIHmG6vGdUJwRzltS1dvUVk). Note that the data is sorted by labels so please shuffle at least the validation/testing data.

## TensorFlow Baselines

We suggest directly get TensorFlow docker image to install [TensorFlow-GPU](https://www.tensorflow.org/install/docker).
For TensorFlow-CPU compiled with AVX2, we recommend using this precompiled [build](https://github.com/lakshayg/tensorflow-build).

Also there is a TensorFlow docker image specifically built for CPUs with AVX-512 instructions, to get it use:

```bash
docker pull clearlinux/stacks-dlrs_2-mkl    
```

`config.py` controls the parameters of TensorFlow training like `learning rate`. `example_full_softmax.py, example_sampled_softmax.py` are example files for `Amazon-670K` dataset with full softmax and sampled softmax respectively.


# Build/Run on Intel platform

## Prerequisites:
CMake >= 3.0
Intel Compiler (ICC) >= 19

## Build with ICC compiler
```
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
cd /path/to/slide-root
mkdir -p bin && cd bin 
# BDW (AVX2)
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc
# SKX/CLX (AVX512)
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc -DOPT_AVX512=1
# CPX (AVX512 + BF16)
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc -DOPT_AVX512=1 -DOPT_AVX512_BF16=1
make -j
```

## Run on Intel SKX/CLX/CPX
```
cd bin
OMP_NUM_THREADS=<num-of-logic-processor> KMP_HW_SUBSET=<num-of-sockets>s,<num-of-cores-per-socket>c,<num-of-logic-thread-per-core>t KMP_AFFINITY=compact,granularity=fine KMP_BLOCKTIME=200 ./runme ../SLIDE/Config_amz.csv
For example, on CLX8280 2Sx28c:
OMP_NUM_THREADS=112 KMP_HW_SUBSET=2s,28c,2t KMP_AFFINITY=compact,granularity=fine KMP_BLOCKTIME=200 ./runme ../SLIDE/Config_amz.csv
```
For best performance please set Batchsize=multiple-of-logic-core-number from SLIDE/Config_amz.csv.

Results can be checked from the log file under dataset:
```
tail -f dataset/log.txt
```
