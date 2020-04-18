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
      _nodeDataOpt[i].active = new bool[_noOfNodes];
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

        generate(_weights, _weights + _noOfNodes * previousLayerNumOfNodes, [&] () { return distribution(dre); });
        generate(_bias, _bias + _noOfNodes, [&] () { return distribution(dre); });


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

    _train_array = new train<T>[noOfNodes*batchsize];

    // create nodes for this layer
#pragma omp parallel for
    for (size_t i = 0; i < noOfNodes; i++)
    {
#if !OPT_IA
        _Nodes[i].Update(previousLayerNumOfNodes, i, _layerID, type, batchsize, _weights+previousLayerNumOfNodes*i,
                _bias[i], _adamAvgMom+previousLayerNumOfNodes*i , _adamAvgVel+previousLayerNumOfNodes*i, _train_array);
        addtoHashTable(_Nodes[i]._weights, previousLayerNumOfNodes, _Nodes[i]._bias, i);
#else
        addtoHashTable(&_weights[previousLayerNumOfNodes * i], previousLayerNumOfNodes, _bias[i], i);
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
void Layer<T>::addtoHashTable(T* weights, int length, T bias, int ID)
{
    //LSH logic
    int *hashes;
    if(HashFunction==1) {
        hashes = _wtaHasher->getHash(weights);
    }else if (HashFunction==2) {
        hashes = _dwtaHasher->getHashEasy(weights, length, TOPK);
    }else if (HashFunction==3) {
        hashes = _MinHasher->getHashEasy(_binids, weights, length, TOPK);
    }else if (HashFunction==4) {
        hashes = _srp->getHash(weights, length);
    }

    int * hashIndices = _hashTables->hashesToIndex(hashes);
    int * bucketIndices = _hashTables->add(hashIndices, ID+1);

#if !OPT_IA
    _Nodes[ID]._indicesInTables = hashIndices;
    _Nodes[ID]._indicesInBuckets = bucketIndices;
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
float innerproduct(int* index1, T* value1, int len1, T* value2){
    float total = 0;
    for (int i=0; i<len1; i++){
        total+=value1[i]*value2[index1[i]];
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
        float tmp = innerproduct(in_indices, in_values, ICI, &_weights[_previousLayerNumOfNodes * s]);
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
  if (_type == NodeType::Softmax)
    _normalizationConstants[inputID] = 0;

  // find activation for all ACTIVE nodes in layer
  for (int oci = 0; oci < OCI; oci++) {
    bool &active = _nodeDataOpt[inputID].active[oci];
    T &grad = _nodeDataOpt[inputID].grads[oci];
    T &value = _nodeDataOpt[inputID].values[oci];
    int oc = _nodeDataOpt[inputID].indices[oci];
    T *w = &_weights[oc * _previousLayerNumOfNodes];

    if (!active) active = true; // initialize active to false;
#define PF_STR 32
    float res = _bias[oc];
    for (int ici = 0; ici < ICI; ici++)
    {
      res += float(w[in_indices[ici]]) * float(in_values[ici]);
      //if (ici + PF_STR < ICI)
      //  _mm_prefetch(&w[in_indices[ici + PF_STR]], _MM_HINT_T0);
    }

    switch (_type)
    {
    case NodeType::ReLU:
      if (res < 0) {
        res = 0;
        grad = 0;
      }
      break;
    case NodeType::Softmax:
      break;
    default:
      cout << "Invalid Node type from Constructor" <<endl;
      break;
    }
    value = res;

    if(_type == NodeType::Softmax && res > maxValue){
      maxValue = res;
    }
  }

  if(_type == NodeType::Softmax) {
    if (std::is_same<T, float>::value) {
      constexpr int V = 16;
      int O = (OCI + V - 1) / V;
      int Vr = OCI % V ? OCI % V : V;
      __m512 vec_max = _mm512_set1_ps(maxValue);
      __m512 vec_sum = _mm512_setzero_ps();
      __m512 vec_zero = _mm512_setzero_ps();
      for (int o = 0; o < O; o++) {
        int Vx = o == O - 1 ? Vr : V;
        __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);

        __m512 vec_val = _mm512_maskz_load_ps(k, &_nodeDataOpt[inputID].values[o * V]);
        vec_val = _mm512_mask_exp_ps(vec_zero, k, vec_val - vec_max);
        vec_sum += vec_val;
        _mm512_mask_storeu_ps(&_nodeDataOpt[inputID].values[o * V], k, vec_val);
      }
      float sum = _mm512_reduce_add_ps(vec_sum);
      _normalizationConstants[inputID] = sum;
    } else {
      for (int oci = 0; oci < OCI; oci++) {
        float v = _nodeDataOpt[inputID].values[oci];
        float realActivation = exp(v - maxValue);
        _nodeDataOpt[inputID].values[oci] = realActivation;
        _normalizationConstants[inputID] += realActivation;
      }
    }
  }

  return in;
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
      delete[] _nodeDataOpt[i].active;
    }
    delete[] _nodeDataOpt;
    delete[] _adamAvgMomBias;
    delete[] _adamAvgVelBias;
    delete[] _weightGrads;
    delete[] _biasGrads;
#endif

    delete[] _adamAvgMom;
    delete[] _adamAvgVel;
    delete[] _Nodes;
    delete[] _weights;
    delete[] _bias;

    delete _wtaHasher;
    delete _dwtaHasher;
    delete _srp;
    delete _MinHasher;
    delete [] _randNode;
    delete[] _train_array;
}

template class Layer<float>;
template class Layer<bfloat16>;
