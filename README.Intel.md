# Build/Run on Intel platform

## Prerequisites:
CMake >= 3.0
Intel Compiler (ICC) >= 19
1G HugeTLB enabled

## Build with ICC compiler
```
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
cd /path/to/slide-root
mkdir -p bin && cd bin 
cmake .. -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_FLAGS=-xCore-AVX512 
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
