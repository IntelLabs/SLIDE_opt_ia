#pragma once
#include "Layer.h"
#include <chrono>
#include "cnpy.h"
#include <sys/mman.h>
#include "DataLayerOpt.h"

using namespace std;

template <class T>
class Network
{
private:
	Layer<T>** _hiddenlayers;
	float _learningRate;
	int _numberOfLayers;
	int* _sizesOfLayers;
	NodeType* _layersTypes;
	float * _Sparsity;
	//int* _inputIDs;
	int  _currentBatchSize;


public:
	Network(int* sizesOfLayers, NodeType* layersTypes, int noOfLayers, int batchsize, float lr, int inputdim, int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr);
	Layer<T>* getLayer(int LayerID);
	int predictClass(int ** inputIndices, T ** inputValues, int * length, int ** labels, int *labelsize);
	int predictClassOpt(DataLayerOpt<T> &dataLayerOpt, size_t batchIndex);
	int ProcessInput(int** inputIndices, T** inputValues, int* lengths, int ** label, int *labelsize, int iter, bool rehash, bool rebuild);
  int ProcessInputOpt(DataLayerOpt<T> &dataLayerOpt, size_t batchIndex, int iter, bool rehash, bool rebuild);
	void saveWeights(string file);
	~Network();
	void * operator new(size_t size){
	    cout << "new Network" << endl;
	    void* ptr = mmap(NULL, size,
	        PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
	        -1, 0);
	    if (ptr == MAP_FAILED){
	        ptr = mmap(NULL, size,
	            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
	            -1, 0);
	    }
	    if (ptr == MAP_FAILED){
	        std::cout << "mmap failed at Network." << std::endl;
	    }
	    return ptr;
	}
	void operator delete(void * pointer){munmap(pointer, sizeof(Network));};
};

