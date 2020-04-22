#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <climits>
#include "Config.h"
#include <bitset>
#include <fstream>
#include <omp.h>
#include <x86intrin.h>
#include "Bfloat16.h"

using namespace std;


template <class T>
Layer<T>::Layer(size_t noOfNodes, int previousLayerNumOfNodes, int layerID, NodeType type, int batchsize,  int K, int L, int RangePow, float Sparsity, T* weights, T* bias, float *adamAvgMom, float *adamAvgVel) {
    _layerID = layerID;
    _noOfNodes = noOfNodes;
#if !OPT_IA
    _Nodes = new Node[noOfNodes];
#endif
    _type = type;
    _noOfActive = floor(_noOfNodes * Sparsity);
    _K = K;
    _L = L;
    _batchsize = batchsize;
    _RangeRow = RangePow;
    _previousLayerNumOfNodes = previousLayerNumOfNodes;

    _weightsOrder = WeightsOrder::OI;
#if OPT_IA
    if (_noOfActive == _noOfNodes) {
      _weightsOrder = WeightsOrder::IO;
    }
#endif

// create a list of random nodes just in case not enough nodes from hashtable for active nodes.
    _randNode = new int[_noOfNodes];
    for (size_t n = 0; n < _noOfNodes; n++) {
        _randNode[n] = n;
    }

    std::random_shuffle(_randNode, _randNode + _noOfNodes);

//TODO: Initialize Hash Tables and add the nodes. Done by Beidi
    _hashTables = new LSH(_K, _L, RangePow);

    if (HashFunction == 1) {
        _wtaHasher = new WtaHash<T>(_K * _L, previousLayerNumOfNodes);
    } else if (HashFunction == 2) {
        _binids = new int[previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash<T>(_K * _L, previousLayerNumOfNodes);
    } else if (HashFunction == 3) {
        _binids = new int[previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash<T>(_K * _L, previousLayerNumOfNodes);
        _MinHasher->getMap(previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {
        _srp = new SparseRandomProjection<T>(previousLayerNumOfNodes, _K * _L, Ratio);
    }

#if OPT_IA
    _nodeDataOpt = new NodeDataOpt[_batchsize];
    for (int i = 0; i < _batchsize; i++) {
      _nodeDataOpt[i].size = _noOfNodes; // assume dense
      _nodeDataOpt[i].indices = new int[_noOfNodes];
      _nodeDataOpt[i].values = new T[_noOfNodes];
      _nodeDataOpt[i].grads = new T[_noOfNodes];
    }
#endif

    if (LOADWEIGHT) {
        _weights = weights;
        _bias = bias;

        if (ADAM){
            _adamAvgMom = adamAvgMom;
            _adamAvgVel = adamAvgVel;
        }

    }else{
        _weights = new T[_noOfNodes * previousLayerNumOfNodes]();
        _bias = new T[_noOfNodes];
#if OPT_IA
        _weightGrads = new T[_noOfNodes * previousLayerNumOfNodes]();
        _biasGrads = new T[_noOfNodes];
#endif
        random_device rd;
        default_random_engine dre(rd());
        normal_distribution<float> distribution(0.0, 0.01);

        generate(_weights, _weights + _noOfNodes * previousLayerNumOfNodes, [&]() { return T(distribution(dre)); });
        generate(_bias, _bias + _noOfNodes, [&] () { return T(distribution(dre)); });

        if (ADAM)
        {
            _adamAvgMom = new float[_noOfNodes * previousLayerNumOfNodes]();
            _adamAvgVel = new float[_noOfNodes * previousLayerNumOfNodes]();

#if OPT_IA
            _adamAvgMomBias = new float[_noOfNodes]();
            _adamAvgVelBias = new float[_noOfNodes]();
#endif
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

#if !OPT_IA
    _train_array = new train<T>[noOfNodes*batchsize];
#endif

    // create nodes for this layer
#pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
#if !OPT_IA
        _Nodes[i].Update(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i,
                _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i, _train_array);
        addtoHashTable(_Nodes[i]._weights, previousLayerNumOfNodes, _Nodes[i]._bias, i);
#else
        if (_weightsOrder == WeightsOrder::OI)
          addtoHashTable(&_weights[previousLayerNumOfNodes * i], previousLayerNumOfNodes, _bias[i], i);
        else
          addtoHashTable(&_weights[i], previousLayerNumOfNodes, _bias[i], i, noOfNodes);
#endif
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout<< noOfNodes<<" "<<1.0 * timeDiffInMiliseconds<<std::endl;

    if (type == NodeType::Softmax)
    {
        _normalizationConstants = new float[batchsize]();
    }
}


template <class T>
void Layer<T>::updateTable()
{

    if (HashFunction == 1) {
         delete _wtaHasher;
        _wtaHasher = new WtaHash<T>(_K * _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 2) {
         delete _dwtaHasher, _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _dwtaHasher = new DensifiedWtaHash<T>(_K * _L, _previousLayerNumOfNodes);
    } else if (HashFunction == 3) {

         delete _MinHasher,  _binids;
        _binids = new int[_previousLayerNumOfNodes];
        _MinHasher = new DensifiedMinhash<T>(_K * _L, _previousLayerNumOfNodes);
        _MinHasher->getMap(_previousLayerNumOfNodes, _binids);
    } else if (HashFunction == 4) {

        _srp = new SparseRandomProjection<T>(_previousLayerNumOfNodes, _K * _L, Ratio);

    }
}


template <class T>
void Layer<T>::updateRandomNodes()
{
    std::random_shuffle(_randNode, _randNode + _noOfNodes);
}


template <class T>
void Layer<T>::addtoHashTable(T* weights, int length, T bias, int ID, int stride)
{
    //LSH logic
    int *hashes;
    if(HashFunction==1) {
        hashes = _wtaHasher->getHash(weights); // TODO: IO
    }else if (HashFunction==2) {
        hashes = _dwtaHasher->getHashEasy(weights, length, TOPK, stride);
    }else if (HashFunction==3) {
        hashes = _MinHasher->getHashEasy(_binids, weights, length, TOPK); // TODO: IO
    }else if (HashFunction==4) {
        hashes = _srp->getHash(weights, length); // TODO: IO
    }

#if !OPT_IA
    int * hashIndices = _hashTables->hashesToIndex(hashes);
    int * bucketIndices = _hashTables->add(hashIndices, ID+1);
    _Nodes[ID]._indicesInTables = hashIndices;
    _Nodes[ID]._indicesInBuckets = bucketIndices;
#else
    _hashTables->hashesToIndexAddOpt(hashes, ID + 1);
#endif

    delete [] hashes;
}


template <class T>
Node<T>* Layer<T>::getNodebyID(size_t nodeID)
{
    assert(("nodeID less than _noOfNodes" , nodeID < _noOfNodes));
    return &_Nodes[nodeID];
}


template <class T>
Node<T>* Layer<T>::getAllNodes()
{
    return _Nodes;
}

template <class T>
int Layer<T>::getNodeCount()
{
    return _noOfNodes;
}

template <class T>
float Layer<T>::getNomalizationConstant(int inputID)
{
    assert(("Error Call to Normalization Constant for non - softmax layer", _type == NodeType::Softmax));
    return _normalizationConstants[inputID];
}


template <class T>
float innerproduct(int* index1, T* value1, int len1, T* value2, int stride = 1){
    float total = 0;
    for (int i = 0; i < len1; i++){
        total += float(value1[i])* float(value2[index1[i] * stride]);
    }
    return total;
}

template <class T>
float collision(int* hashes, int* table_hashes, int k, int l){
    int cp = 0;
    for (int i=0; i<l; i=i+k){
        int tmp = 0;
        for (int j=i; j< i+k;j++){
            if(hashes[j]==table_hashes[j]){
                tmp++;
            }
        }
        if (tmp==k){
            cp++;
        }
    }
    return cp*1.0/(l/k);
}

template <class T>
int Layer<T>::queryActiveNodeandComputeActivations(int** activenodesperlayer, T** activeValuesperlayer, int* lengths, int layerIndex, int inputID, int* label, int labelsize, float Sparsity, int iter)
{
    //LSH QueryLogic

    //Beidi. Query out all the candidate nodes
    int len;
    int in = 0;

    if(Sparsity == 1.0){
        len = _noOfNodes;
        lengths[layerIndex + 1] = len;
        activenodesperlayer[layerIndex + 1] = new int[len]; //assuming not intitialized;
        for (int i = 0; i < len; i++)
        {
            activenodesperlayer[layerIndex + 1][i] = i;
        }
    }
    else
    {
        if (Mode==1) {
            int *hashes;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);

            // Get candidates from hashtable
            auto t00 = std::chrono::high_resolution_clock::now();

            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        counts[label[i]] = _L;
                    }
                }
            }

            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            auto t11 = std::chrono::high_resolution_clock::now();

            //thresholding
            auto t3 = std::chrono::high_resolution_clock::now();
            vector<int> vect;
            for (auto &&x : counts){
                if (x.second>THRESH){
                    vect.push_back(x.first);
                }
            }

            len = vect.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = vect[i];
            }
            auto t33 = std::chrono::high_resolution_clock::now();
            in = len;

            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        if (Mode==4) {
            int *hashes;
            if (HashFunction == 1) {
                hashes = _wtaHasher->getHash(activeValuesperlayer[layerIndex]);
            } else if (HashFunction == 2) {
                hashes = _dwtaHasher->getHash(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                              lengths[layerIndex]);
            } else if (HashFunction == 3) {
                hashes = _MinHasher->getHashEasy(_binids, activeValuesperlayer[layerIndex], lengths[layerIndex], TOPK);
            } else if (HashFunction == 4) {
                hashes = _srp->getHashSparse(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex]);
            }
            int *hashIndices = _hashTables->hashesToIndex(hashes);
            int **actives = _hashTables->retrieveRaw(hashIndices);
            // we now have a sparse array of indices of active nodes

            // Get candidates from hashtable
            std::map<int, size_t> counts;
            // Make sure that the true label node is in candidates
            if (_type == NodeType::Softmax && labelsize > 0) {
                for (int i = 0; i < labelsize ;i++){
                    counts[label[i]] = _L;
                }
            }

            for (int i = 0; i < _L; i++) {
                if (actives[i] == NULL) {
                    continue;
                } else {
                    // copy sparse array into (dense) map
                    for (int j = 0; j < BUCKETSIZE; j++) {
                        int tempID = actives[i][j] - 1;
                        if (tempID >= 0) {
                            counts[tempID] += 1;
                        } else {
                            break;
                        }
                    }
                }
            }

            in = counts.size();
            if (counts.size()<1500){
                srand(time(NULL));
                size_t start = rand() % _noOfNodes;
                for (size_t i = start; i < _noOfNodes; i++) {
                    if (counts.size() >= 1000) {
                        break;
                    }
                    if (counts.count(_randNode[i]) == 0) {
                        counts[_randNode[i]] = 0;
                    }
                }

                if (counts.size() < 1000) {
                    for (size_t i = 0; i < _noOfNodes; i++) {
                        if (counts.size() >= 1000) {
                            break;
                        }
                        if (counts.count(_randNode[i]) == 0) {
                            counts[_randNode[i]] = 0;
                        }
                    }
                }
            }

            len = counts.size();
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            // copy map into new array
            int i=0;
            for (auto &&x : counts) {
                activenodesperlayer[layerIndex + 1][i] = x.first;
                i++;
            }

            delete[] hashes;
            delete[] hashIndices;
            delete[] actives;

        }
        else if (Mode == 2 & _type== NodeType::Softmax) {
            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];

            auto t1 = std::chrono::high_resolution_clock::now();
            bitset <MAPLEN> bs;
            int tmpsize = 0;
            if (_type == NodeType::Softmax) {
                if (labelsize > 0) {
                    for (int i=0; i<labelsize ;i++){
                        activenodesperlayer[layerIndex + 1][i] = label[i];
                        bs[label[i]] = 1;
                    }
                    tmpsize = labelsize;
                }
            }

            while(tmpsize<len){
                int v = rand() % _noOfNodes;
                if(bs[v]==0) {
                    activenodesperlayer[layerIndex + 1][tmpsize] = v;
                    bs[v]=1;
                    tmpsize++;
                }
            }



            auto t2 = std::chrono::high_resolution_clock::now();
//            std::cout << "sampling "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

        }

        else if (Mode==3 & _type== NodeType::Softmax){

            len = floor(_noOfNodes * Sparsity);
            lengths[layerIndex + 1] = len;
            activenodesperlayer[layerIndex + 1] = new int[len];
            vector<pair<float, int> > sortW;
            int what = 0;

            for (size_t s = 0; s < _noOfNodes; s++) {
                float tmp = innerproduct(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex],
                                         lengths[layerIndex], _Nodes[s]._weights);
                tmp += _Nodes[s]._bias;
                if (find(label, label + labelsize, s) != label + labelsize) {
                    sortW.push_back(make_pair(-1000000000, s));
                    what++;
                }
                else{
                    sortW.push_back(make_pair(-tmp, s));
                }
            }

            std::sort(begin(sortW), end(sortW));

            for (int i = 0; i < len; i++) {
                activenodesperlayer[layerIndex + 1][i] = sortW[i].second;
                if (find (label, label+labelsize, sortW[i].second)!= label+labelsize){
                    in=1;
                }
            }
        }
    }

    //***********************************
    activeValuesperlayer[layerIndex + 1] = new T[len]; //assuming its not initialized else memory leak;
    T maxValue = 0;
    if (_type == NodeType::Softmax)
        _normalizationConstants[inputID] = 0;

    // find activation for all ACTIVE nodes in layer
    for (int i = 0; i < len; i++)
    {
        activeValuesperlayer[layerIndex + 1][i] = _Nodes[activenodesperlayer[layerIndex + 1][i]].getActivation(activenodesperlayer[layerIndex], activeValuesperlayer[layerIndex], lengths[layerIndex], inputID);
        if(_type == NodeType::Softmax && activeValuesperlayer[layerIndex + 1][i] > maxValue){
            maxValue = activeValuesperlayer[layerIndex + 1][i];
        }
    }

    if(_type == NodeType::Softmax) {
      if (std::is_same<T, float>::value) {
        constexpr int V = 16;
        int O = (len + V - 1) / V;
        int Vr = len % V ? len % V : V;
        __m512 vec_max = _mm512_set1_ps(maxValue);
        __m512 vec_sum = _mm512_setzero_ps();
        __m512 vec_zero = _mm512_setzero_ps();
        for (int o = 0; o < O; o++) {
            int Vx = o == O - 1 ? Vr : V;
            __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);

            __m512 vec_val = _mm512_maskz_load_ps(k, &activeValuesperlayer[layerIndex + 1][o * V]);
            vec_val = _mm512_mask_exp_ps(vec_zero, k, vec_val - vec_max);
            vec_sum += vec_val;
            _mm512_mask_storeu_ps(&activeValuesperlayer[layerIndex + 1][o * V], k, vec_val);
            for (int v = 0; v < Vx; v++) {
                float val = activeValuesperlayer[layerIndex + 1][o * V + v];
                _Nodes[activenodesperlayer[layerIndex + 1][o * V + v]].SetlastActivation(inputID, val);
            }
        }
        float sum = _mm512_reduce_add_ps(vec_sum);
        _normalizationConstants[inputID] = sum;
      } else {
        for (int i = 0; i < len; i++) {
            T realActivation = exp(activeValuesperlayer[layerIndex + 1][i] - maxValue);
            activeValuesperlayer[layerIndex + 1][i] = realActivation;
            _Nodes[activenodesperlayer[layerIndex + 1][i]].SetlastActivation(inputID, realActivation);
            _normalizationConstants[inputID] += realActivation;
        }
      }
    }

    return in;
}


template <class T>
int Layer<T>::queryActiveNodeandComputeActivationsOpt(
    int* in_indices, T* in_values, int ICI,
    int layerID, int inputID, int* label, int labelsize,
    float Sparsity, int iter) {

  //LSH QueryLogic

  //Beidi. Query out all the candidate nodes
  int OCI;
  int in = 0;

  if (Sparsity == 1.0){
    OCI = _noOfNodes;
    _nodeDataOpt[inputID].size = OCI;
    for (int i = 0; i < OCI; i++) {
      _nodeDataOpt[inputID].indices[i] = i;
    }
  } else {
    if (Mode==1) {
      int *hashes;
      if (HashFunction == 1) {
        hashes = _wtaHasher->getHash(in_values);
      } else if (HashFunction == 2) {
        hashes = _dwtaHasher->getHash(in_indices, in_values, ICI);
      } else if (HashFunction == 3) {
        hashes = _MinHasher->getHashEasy(_binids, in_values, ICI, TOPK);
      } else if (HashFunction == 4) {
        hashes = _srp->getHashSparse(in_indices, in_values, ICI);
      }
      int *hashIndices = _hashTables->hashesToIndex(hashes);
      int **actives = _hashTables->retrieveRaw(hashIndices);

      // Get candidates from hashtable
      auto t00 = std::chrono::high_resolution_clock::now();

      std::map<int, size_t> counts;
      // Make sure that the true label node is in candidates
      if (_type == NodeType::Softmax) {
        if (labelsize > 0) {
          for (int i=0; i<labelsize ;i++){
            counts[label[i]] = _L;
          }
        }
      }

      for (int i = 0; i < _L; i++) {
        if (actives[i] == NULL) {
          continue;
        } else {
          for (int j = 0; j < BUCKETSIZE; j++) {
            int tempID = actives[i][j] - 1;
            if (tempID >= 0) {
              counts[tempID] += 1;
            } else {
              break;
            }
          }
        }
      }
      auto t11 = std::chrono::high_resolution_clock::now();

      //thresholding
      auto t3 = std::chrono::high_resolution_clock::now();
      vector<int> vect;
      for (auto &&x : counts){
        if (x.second>THRESH){
          vect.push_back(x.first);
        }
      }

      OCI = vect.size();
      _nodeDataOpt[inputID].size = OCI;

      for (int i = 0; i < OCI; i++) {
        _nodeDataOpt[inputID].indices[i] = vect[i];
      }
      auto t33 = std::chrono::high_resolution_clock::now();
      in = OCI;

      delete[] hashes;
      delete[] hashIndices;
      delete[] actives;

    }
    if (Mode==4) {
      int *hashes;
      if (HashFunction == 1) {
        hashes = _wtaHasher->getHash(in_values);
      } else if (HashFunction == 2) {
        hashes = _dwtaHasher->getHash(in_indices, in_values, ICI);
      } else if (HashFunction == 3) {
        hashes = _MinHasher->getHashEasy(_binids, in_values, ICI, TOPK);
      } else if (HashFunction == 4) {
        hashes = _srp->getHashSparse(in_indices, in_values, ICI);
      }
      int *hashIndices = _hashTables->hashesToIndex(hashes);
      int **actives = _hashTables->retrieveRaw(hashIndices);
      // we now have a sparse array of indices of active nodes

      // Get candidates from hashtable
      std::map<int, size_t> counts;
      // Make sure that the true label node is in candidates
      if (_type == NodeType::Softmax && labelsize > 0) {
        for (int i = 0; i < labelsize ;i++){
          counts[label[i]] = _L;
        }
      }

      for (int i = 0; i < _L; i++) {
        if (actives[i] == NULL) {
          continue;
        } else {
          // copy sparse array into (dense) map
          for (int j = 0; j < BUCKETSIZE; j++) {
            int tempID = actives[i][j] - 1;
            if (tempID >= 0) {
              counts[tempID] += 1;
            } else {
              break;
            }
          }
        }
      }

      in = counts.size();
      if (counts.size()<1500){
        srand(time(NULL));
        size_t start = rand() % _noOfNodes;
        for (size_t i = start; i < _noOfNodes; i++) {
          if (counts.size() >= 1000) {
            break;
          }
          if (counts.count(_randNode[i]) == 0) {
            counts[_randNode[i]] = 0;
          }
        }

        if (counts.size() < 1000) {
          for (size_t i = 0; i < _noOfNodes; i++) {
            if (counts.size() >= 1000) {
              break;
            }
            if (counts.count(_randNode[i]) == 0) {
              counts[_randNode[i]] = 0;
            }
          }
        }
      }

      OCI = counts.size();
      _nodeDataOpt[inputID].size = OCI;

      // copy map into new array
      int i=0;
      for (auto &&x : counts) {
        _nodeDataOpt[inputID].indices[i] = x.first;
        i++;
      }

      delete[] hashes;
      delete[] hashIndices;
      delete[] actives;

    }
    else if (Mode == 2 & _type== NodeType::Softmax) {
      OCI = floor(_noOfNodes * Sparsity);
      _nodeDataOpt[inputID].size = OCI;

      auto t1 = std::chrono::high_resolution_clock::now();
      bitset <MAPLEN> bs;
      int tmpsize = 0;
      if (_type == NodeType::Softmax) {
        if (labelsize > 0) {
          for (int i=0; i<labelsize ;i++){
            _nodeDataOpt[inputID].indices[i] = label[i];
            bs[label[i]] = 1;
          }
          tmpsize = labelsize;
        }
      }

      while(tmpsize < OCI){
        int v = rand() % _noOfNodes;
        if(bs[v]==0) {
          _nodeDataOpt[inputID].indices[tmpsize] = v;
          bs[v]=1;
          tmpsize++;
        }
      }



      auto t2 = std::chrono::high_resolution_clock::now();
      //            std::cout << "sampling "<<" takes" << 1.0 * timeDiffInMiliseconds << std::endl;

    }

    else if (Mode==3 & _type== NodeType::Softmax){

      OCI = floor(_noOfNodes * Sparsity);
      _nodeDataOpt[inputID].size = OCI;
      vector<pair<float, int> > sortW;
      int what = 0;

      for (size_t s = 0; s < _noOfNodes; s++) {
        float tmp;
        if (_weightsOrder == WeightsOrder::OI)
          tmp = innerproduct(in_indices, in_values, ICI, &_weights[_previousLayerNumOfNodes * s]);
        else
          tmp = innerproduct(in_indices, in_values, ICI, &_weights[s], _noOfNodes);
        tmp += _bias[s];
        if (find(label, label + labelsize, s) != label + labelsize) {
          sortW.push_back(make_pair(-1000000000, s));
          what++;
        }
        else{
          sortW.push_back(make_pair(-tmp, s));
        }
      }

      std::sort(begin(sortW), end(sortW));

      for (int i = 0; i < OCI; i++) {
        _nodeDataOpt[inputID].indices[i] = sortW[i].second;
        if (find (label, label+labelsize, sortW[i].second)!= label+labelsize){
          in=1;
        }
      }
    }
  }

  //***********************************
  float maxValue = 0;
  int IC = _previousLayerNumOfNodes;
  int OC = _noOfNodes;

#if OPT_VEC512
  constexpr int V = 16;
  if (_weightsOrder == WeightsOrder::IO && OC == OCI) {
    int O2 = (OCI + V - 1) / V;
    int Vr = OCI % V ? OCI % V : V;
    __m512 vec_max = _mm512_setzero_ps();
    __m512 vec_zero = _mm512_setzero_ps();
    for (int o2 = 0; o2 < O2; o2++) {
      int Vx = o2 == O2 - 1 ? Vr : V;
      __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
      __m512 vec_out;
      if (std::is_same<float, T>::value) {
        vec_out = _mm512_maskz_load_ps(k, &_bias[o2 * V]);
      } else {
        // TODO: bf16
      }
      for (int ici = 0; ici < ICI; ici++) {
        int ic = in_indices[ici];
        __m512 vec_wei;
        if (std::is_same<float, T>::value) {
          vec_wei = _mm512_maskz_load_ps(k, &_weights[ic * OC + o2 * V]);
        } else {
          // TODO: bf16
        }
        float in = in_values[ici];
        __m512 vec_in = _mm512_set1_ps(in);
        vec_out += vec_in * vec_wei;
      }
      if (_type == NodeType::ReLU) {
        vec_out = _mm512_max_ps(vec_out, vec_zero);
      } else if (_type == NodeType::Softmax) {
        vec_max = _mm512_max_ps(vec_out, vec_max);
      }
      if (std::is_same<float, T>::value) {
        _mm512_mask_storeu_ps(&_nodeDataOpt[inputID].values[o2 * V],
                              k, vec_out);
        //_mm512_mask_storeu_ps(&_nodeDataOpt[inputID].grads[o2 * V],
        //                      k, vec_zero);

      } else {
        // TODO: bf16
      }
    }
    if (_type == NodeType::Softmax)
      maxValue = _mm512_reduce_max_ps(vec_max);
  } else if (_weightsOrder == WeightsOrder::OI && ICI == IC) {
    int I2 = (ICI + V - 1) / V;
    int Vr = ICI % V ? ICI % V : V;
    __m512 vec_max = _mm512_setzero_ps();
    __m512 vec_zero = _mm512_setzero_ps();
    // TODO: unroll and save input load
    for (int oci = 0; oci < OCI; oci++) {
      T &value = _nodeDataOpt[inputID].values[oci];
      int oc = _nodeDataOpt[inputID].indices[oci];

      __m512 vec_out = _mm512_setzero_ps();
      float res = _bias[oc];
      for (int i2 = 0; i2 < I2; i2++) {
        int Vx = i2 == I2 - 1 ? Vr : V;
        __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
        __m512 vec_wei, vec_in;
        if (std::is_same<float, T>::value) {
          vec_wei = _mm512_maskz_load_ps(k, &_weights[oc * IC + i2 * V]);
          vec_in = _mm512_maskz_load_ps(k, &in_values[i2 * V]);
        } else {
          // TODO: bf16
        }
        vec_out += vec_in * vec_wei;
      }
      res = _mm512_reduce_add_ps(vec_out);
      if (_type == NodeType::ReLU) {
        if (res < 0) res = 0;
      } else if (_type == NodeType::Softmax) {
        if (res > maxValue) maxValue = res;
      }
      value = res;
    }
  } else
#endif
  {
    // find activation for all ACTIVE nodes in layer
    for (int oci = 0; oci < OCI; oci++) {
      T &value = _nodeDataOpt[inputID].values[oci];
      int oc = _nodeDataOpt[inputID].indices[oci];

      float res = _bias[oc];
      if (_weightsOrder == WeightsOrder::OI) {
        for (int ici = 0; ici < ICI; ici++) {
          int ic = in_indices[ici];
          res += float(_weights[oc * IC + ic]) * float(in_values[ici]);
        }
      } else {
        for (int ici = 0; ici < ICI; ici++) {
          int ic = in_indices[ici];
          res += float(_weights[ic * OC + oc]) * float(in_values[ici]);
        }
      }
      if (_type == NodeType::ReLU) {
        if (res < 0) res = 0;
      } else if (_type == NodeType::Softmax) {
        if (res > maxValue) maxValue = res;
      }
      value = res;
    }
  }

  if(_type == NodeType::Softmax) {
#if OPT_VEC512
    constexpr int V = 16;
    int O2 = (OCI + V - 1) / V;
    int Vr = OCI % V ? OCI % V : V;
    __m512 vec_max = _mm512_set1_ps(maxValue);
    __m512 vec_sum = _mm512_setzero_ps();
    __m512 vec_zero = _mm512_setzero_ps();
    for (int o2 = 0; o2 < O2; o2++) {
      int Vx = o2 == O2 - 1 ? Vr : V;
      __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
      __m512 vec_val;
      if (std::is_same<T, float>::value) {
        vec_val = _mm512_maskz_load_ps(k, &_nodeDataOpt[inputID].values[o2 * V]);
      } else { // bfloat16
        __m256i vec256_zero = _mm256_setzero_si256();
        vec_val = _mm512_mask_load_bf16_as_fp32(vec256_zero, k, &_nodeDataOpt[inputID].values[o2 * V]);
      }
      vec_val = _mm512_mask_exp_ps(vec_zero, k, vec_val - vec_max);
      vec_sum += vec_val;
      if (std::is_same<T, float>::value) {
        _mm512_mask_storeu_ps(&_nodeDataOpt[inputID].values[o2 * V], k, vec_val);
      } else {
        __m256i vec256_val = _mm512_cvt_fp32_to_bf16(vec_val);
        _mm256_mask_storeu_epi16(&_nodeDataOpt[inputID].values[o2 * V], k, vec256_val);
      }
    }
    float sum = _mm512_reduce_add_ps(vec_sum);
    _normalizationConstants[inputID] = sum;
#else
    _normalizationConstants[inputID] = 0;
    for (int oci = 0; oci < OCI; oci++) {
      float v = _nodeDataOpt[inputID].values[oci];
      float realActivation = exp(v - maxValue);
      _nodeDataOpt[inputID].values[oci] = realActivation;
      _normalizationConstants[inputID] += realActivation;
    }
#endif
  }

  return in;
}

template <class T>
void Layer<T>::backPropagateFirstLayerOpt(DataLayerOpt<T> &dataLayerOpt,
                                          int inputID,
                                          int recordIndex,
                                          float tmplr) {
  int OCI = _nodeDataOpt[inputID].size;
  int ICI = dataLayerOpt.lengthByRecordIndex(recordIndex);
  int OC = _noOfNodes;
  int IC = _previousLayerNumOfNodes;
  bool isOIWeights = _weightsOrder == WeightsOrder::OI;

#if OPT_VEC512
  if (!isOIWeights && ADAM) {
    constexpr int V = 16;
    int O2 = (OCI + V - 1) / V;
    int Vr = OCI % V ? OCI % V : V;
    int *prevIndices = dataLayerOpt.indicesByRecordIndex(recordIndex);
    T *prevValues = dataLayerOpt.valuesByRecordIndex(recordIndex);
    __m512 vec_zero = _mm512_setzero_ps();
    for (int ici = 0; ici < ICI; ici++) {
      int ic = prevIndices[ici];
      float x = prevValues[ici];
      __m512 vec_x = _mm512_set1_ps(x);
      for (int o2 = 0; o2 < O2; o2++) {
        int Vx = o2 == O2 - 1 ? Vr : V;
        __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
        __m512 vec_gy, vec_gw, vec_gb;
        if (std::is_same<float, T>::value) {
          vec_gy = _mm512_maskz_load_ps(k, &_nodeDataOpt[inputID].grads[o2 * V]);
          vec_gw = _mm512_maskz_load_ps(k, &_weightGrads[ic * OC + o2 * V]);
          if (ici == ICI - 1)
            vec_gb = _mm512_maskz_load_ps(k, &_biasGrads[o2 * V]);
        } else {
          // TODO: bf16
        }
        vec_gw += vec_gy * vec_x;
        if (ici == 0)
          vec_gb += vec_gy;
        if (std::is_same<float, T>::value) {
          _mm512_mask_storeu_ps(&_weightGrads[ic * OC + o2 * V], k, vec_gw);
          if (ici == 0) {
            _mm512_mask_storeu_ps(&_biasGrads[o2 * V], k, vec_gb);
          }
        } else {
          // TODO: bf16
        }
      }
    }
  } else
#endif
  {
    for (int oci = 0; oci < OCI; oci++) {
      int oc = _nodeDataOpt[inputID].indices[oci];
      T &grad = _nodeDataOpt[inputID].grads[oci];
      T *prevValues = dataLayerOpt.valuesByRecordIndex(recordIndex);
      T &gb = _biasGrads[oc];

      for (int ici = 0; ici < ICI; ici++) {
        int ic = dataLayerOpt.indicesByRecordIndex(recordIndex)[ici];
        int idx = isOIWeights ? IC * oc + ic : ic * OC + oc;
        T &gw = _weightGrads[idx];

        T grad_t = grad * prevValues[ici];
        if (ADAM) {
          gw += grad_t;
        } else {
          gw += tmplr * grad_t;
        }
      }
      if (ADAM) {
        T biasgrad_t = grad;
        gb += biasgrad_t;
      } else {
        gb += tmplr * grad;
      }
    }
  }
}

template <class T>
void Layer<T>::backPropagateOpt(Layer<T> *prev_layer, int inputID, float tmplr) {
  int OCI = _nodeDataOpt[inputID].size;
  int ICI = prev_layer->_nodeDataOpt[inputID].size;
  int OC = _noOfNodes;
  int IC = _previousLayerNumOfNodes;
  bool isOIWeights = _weightsOrder == WeightsOrder::OI;
  T *prevValues = prev_layer->_nodeDataOpt[inputID].values;
  T *prevGrads = prev_layer->_nodeDataOpt[inputID].grads;

#if OPT_VEC512
  constexpr int V = 16;
  int I2 = (ICI + V - 1) / V;
  int Vr = ICI % V ? ICI % V : V;

  if (isOIWeights && ICI == IC && ADAM) {
    for (int oci = 0; oci < OCI; oci++) {
      int oc = _nodeDataOpt[inputID].indices[oci];
      float gy = _nodeDataOpt[inputID].grads[oci];
      float gb = _biasGrads[oc];
      gb += gy;
      _biasGrads[oc] = gb;

      for (int i2 = 0; i2 < I2; i2++) {
        int Vx = i2 == I2 - 1 ? Vr : V;
        __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
        __m512 vec_gx = _mm512_setzero_ps();
        __m512 vec_zero = _mm512_setzero_ps();
        __m512 vec_x;
        if (std::is_same<float, T>::value) {
          vec_x = _mm512_maskz_load_ps(k, &prevValues[i2 * V]);
          if (oci > 0) {
            vec_gx = _mm512_maskz_load_ps(k, &prevGrads[i2 * V]);
          }
        } else {
          // TODO: bf16
        }
        __mmask16 mask = _mm512_mask_cmp_ps_mask(k, vec_zero, vec_x, _MM_CMPINT_LT);

        int idx = IC * oc + i2 * V;
        __m512 vec_gy = _mm512_set1_ps(gy);
        __m512 vec_w, vec_gw;
        if (std::is_same<float, T>::value) {
          vec_w = _mm512_maskz_load_ps(k, &_weights[idx]);
          vec_gw = _mm512_maskz_load_ps(k, &_weightGrads[idx]);
        } else {
          // TODO: bf16
        }
        vec_gx = _mm512_mask3_fmadd_ps(vec_gy, vec_w, vec_gx, mask);
        //vec_gx += vec_gy * vec_w; // gx = gy * w
        vec_gw += vec_gy * vec_x; // gw = gy * x
        if (std::is_same<float, T>::value) {
          _mm512_mask_storeu_ps(&_weightGrads[idx], k, vec_gw);
          _mm512_mask_storeu_ps(&prevGrads[i2 * V], k, vec_gx);
        } else {
          // TODO: bf16
        }

      }
    }
  } else
#endif
  {
    for (int oci = 0; oci < OCI; oci++) {
      int oc = _nodeDataOpt[inputID].indices[oci];
      T &grad = _nodeDataOpt[inputID].grads[oci];
      T &gb = _biasGrads[oc];

      for (int ici = 0; ici < ICI; ici++) {
        int ic = prev_layer->_nodeDataOpt[inputID].indices[ici];
        int idx = isOIWeights ? IC * oc + ic : ic * OC + oc;
        T w = _weights[idx];
        T &gw = _weightGrads[idx];
        if (oci == 0)
          prevGrads[ici] = 0;
        if (prevValues[ici] > 0) {
          prevGrads[ici] += grad * w;
        }
        T grad_t = grad * prevValues[ici];
        if (ADAM) {
          gw += grad_t;
        } else {
          gw += tmplr * grad_t;
        }
      }
      if (ADAM) {
        T biasgrad_t = grad;
        gb += biasgrad_t;
      } else {
        gb += tmplr * grad;
      }
    }
  }
}

template <class T>
void Layer<T>::computeExtraStatsForSoftMaxOpt(int *labels,
                                              int labelSize,
                                              int inputID,
                                              int currentBatchSize) {
  int OCI = _nodeDataOpt[inputID].size;
#if OPT_VEC512
  constexpr int V = 16;
  int O2 = (OCI + V - 1) / V;
  int Vr = OCI % V ? OCI % V : V;
  __m512 vec_batchsize = _mm512_set1_ps(-currentBatchSize);
  __m512 vec_bias = _mm512_set1_ps(1.0f / labelSize/currentBatchSize);
  __m512 vec_rnorm = _mm512_set1_ps(1.0f / (getNomalizationConstant(inputID) + 0.0000001));
  for (int o2 = 0; o2 < O2; o2++) {
    int Vx = o2 == O2 - 1 ? Vr : V;
    __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
    __m512i vec_oc = _mm512_maskz_load_epi32(k, &_nodeDataOpt[inputID].indices[o2 * V]);
    __m512 vec_value, vec_grad;
    if (std::is_same<float, T>::value) {
      vec_value = _mm512_maskz_load_ps(k, &_nodeDataOpt[inputID].values[o2 * V]);
    } else {
      // TODO: bf16
    }
    vec_value *= vec_rnorm;
    vec_grad = vec_value / vec_batchsize;
    __mmask16 mask = _mm512_int2mask(0);
    for (int i = 0; i < labelSize; i++) {
      __m512i vec_label = _mm512_set1_epi32(labels[i]);
      __mmask16 tmp = _mm512_mask_cmp_epi32_mask(k, vec_oc, vec_label, _MM_CMPINT_EQ);
      mask = _mm512_kor(mask, tmp);
    }
    vec_grad = _mm512_mask_add_ps(vec_grad, mask, vec_grad, vec_bias);
    if (std::is_same<float, T>::value) {
      _mm512_mask_storeu_ps(&_nodeDataOpt[inputID].grads[o2 * V], k, vec_grad);
    } else {
      // TODO: bf16
    }
  }
#else
  for (int oci = 0; oci < OCI; oci++) {
    int oc = _nodeDataOpt[inputID].indices[oci];
    T &value = _nodeDataOpt[inputID].values[oci];
    T &grad = _nodeDataOpt[inputID].grads[oci];

    value /= getNomalizationConstant(inputID) + 0.0000001;
    if (find(labels, labels + labelSize, oc)!= labels + labelSize) {
      grad = (1.0/labelSize - value) / currentBatchSize;
    } else {
      grad = (-value) / currentBatchSize;
    }
  }
#endif
}

template <class T>
void Layer<T>::saveWeights(string file)
{
    if (_layerID==0) {
        cnpy::npz_save(file, "w_layer_0", _weights, {_noOfNodes, _previousLayerNumOfNodes}, "w");
        cnpy::npz_save(file, "b_layer_0", _bias, {_noOfNodes}, "a");
        cnpy::npz_save(file, "am_layer_0", _adamAvgMom, {_noOfNodes, _previousLayerNumOfNodes}, "a");
        cnpy::npz_save(file, "av_layer_0", _adamAvgVel, {_noOfNodes, _previousLayerNumOfNodes}, "a");
        cout<<"save for layer 0"<<endl;
        cout<< float(_weights[0]) <<" "<< float(_weights[1]) <<endl;
    }else{
        cnpy::npz_save(file, "w_layer_"+ to_string(_layerID), _weights, {_noOfNodes, _previousLayerNumOfNodes}, "a");
        cnpy::npz_save(file, "b_layer_"+ to_string(_layerID), _bias, {_noOfNodes}, "a");
        cnpy::npz_save(file, "am_layer_"+ to_string(_layerID), _adamAvgMom, {_noOfNodes, _previousLayerNumOfNodes}, "a");
        cnpy::npz_save(file, "av_layer_"+ to_string(_layerID), _adamAvgVel, {_noOfNodes, _previousLayerNumOfNodes}, "a");
        cout<<"save for layer "<<to_string(_layerID)<<endl;
        cout<< float(_weights[0]) <<" "<< float(_weights[1]) <<endl;
    }
}


template <class T>
Layer<T>::~Layer()
{

    for (size_t i = 0; i < _noOfNodes; i++)
    {
        if (_type == NodeType::Softmax)
        {
            delete[] _normalizationConstants;
        }
    }

#if OPT_IA
    for (int i = 0; i < _batchsize; i++) {
      delete[] _nodeDataOpt[i].values;
      delete[] _nodeDataOpt[i].indices;
      delete[] _nodeDataOpt[i].grads;
    }
    delete[] _nodeDataOpt;
    delete[] _adamAvgMomBias;
    delete[] _adamAvgVelBias;
    delete[] _weightGrads;
    delete[] _biasGrads;
#endif

    delete[] _adamAvgMom;
    delete[] _adamAvgVel;
    delete[] _weights;
    delete[] _bias;

    delete _wtaHasher;
    delete _dwtaHasher;
    delete _srp;
    delete _MinHasher;
    delete [] _randNode;
#if !OPT_IA
    delete[] _Nodes;
    delete[] _train_array;
#endif
}

template class Layer<float>;
template class Layer<bfloat16>;
