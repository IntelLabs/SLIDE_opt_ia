#pragma once
#include "Node.h"
#include "WtaHash.h"
#include "DensifiedMinhash.h"
#include "srp.h"
#include "LSH.h"
#include "DensifiedWtaHash.h"
#include "cnpy.h"
#include <sys/mman.h>

using namespace std;

template <class T>
class Layer
{
private:
	NodeType _type;
	Node<T>* _Nodes;
	int * _randNode;
	float* _normalizationConstants;
  int _K, _L, _RangeRow, _batchsize;
  train<T>* _train_array;


public:
	int _layerID, _noOfActive;
  size_t _previousLayerNumOfNodes;
	size_t _noOfNodes;
	T* _weights;
  T* _weightGrads;
	float* _adamAvgMom;
	float* _adamAvgVel;
	T* _bias;
  T* _biasGrads;
  float* _adamAvgMomBias;
  float* _adamAvgVelBias;

  struct NodeDataOpt {
    int *indices = nullptr;
    T *values = nullptr;
    T *grads = nullptr;
    bool *active = nullptr;
    int size = 0;
  };
  NodeDataOpt *_nodeDataOpt; // per each record

	LSH *_hashTables;
	WtaHash<T> *_wtaHasher;
  DensifiedMinhash<T> *_MinHasher;
  SparseRandomProjection<T> *_srp;
  DensifiedWtaHash<T> *_dwtaHasher;
	int * _binids;
	Layer(size_t _numNodex, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize, int K, int L, int RangePow, float Sparsity, T* weights=NULL, T* bias=NULL, float *adamAvgMom=NULL, float *adamAvgVel=NULL);
	Node<T>* getNodebyID(size_t nodeID);
	Node<T>* getAllNodes();
	int getNodeCount();
	void addtoHashTable(T* weights, int length, T bias, int id);
	float getNomalizationConstant(int inputID);
	int queryActiveNodeandComputeActivations(int** activenodesperlayer, T** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
	int queryActiveNodeandComputeActivationsOpt(int* in_indices, T* in_values, int ICI, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int queryActiveNodes(int** activenodesperlayer, T** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeActivations(int** activenodesperlayer, T** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
    int computeSoftmax(int** activenodesperlayer, T** activeValuesperlayer, int* inlenght, int layerID, int inputID,  int* label, int labelsize, float Sparsity, int iter);
	void saveWeights(string file);
	void updateTable();
	void updateRandomNodes();

	~Layer();

    void * operator new(size_t size){
        cout << "new Layer" << endl;
        void* ptr = mmap(NULL, size,
            PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0);
        if (ptr == MAP_FAILED){
            ptr = mmap(NULL, size,
                PROT_READ | PROT_EXEC | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0);
        }
        if (ptr == MAP_FAILED)
            std::cout << "mmap fail! No new layer!" << std::endl;
        return ptr;};
    void operator delete(void * pointer){munmap(pointer, sizeof(Layer));};

};
