#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <omp.h>
#define DEBUG 1
using namespace std;


Network::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, int inputdim,  int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr) {

    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = lr;
    _currentBatchSize = batchSize;
    _Sparsity = Sparsity;


    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            float* weight, *bias, *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], _currentBatchSize,  K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        } else {

            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            float* weight, *bias, *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<float>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<float>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }
            _hiddenlayers[i] = new Layer(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        }
    }
    cout << "after layer" << endl;
}


Layer *Network::getLayer(int LayerID) {
    if (LayerID < _numberOfLayers)
        return _hiddenlayers[LayerID];
    else {
        cout << "LayerID out of bounds" << endl;
        //TODO:Handle
    }
}

int Network::predictClassOpt(DataLayerOpt &dataLayerOpt, size_t batchIndex) {
  alignas(64) int correctPred = 0;

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+:correctPred)
  for (int n = 0; n < _currentBatchSize; n++) {
    int ICI;
    int *in_indices;
    float *in_values;
    size_t recordIndex = batchIndex * _currentBatchSize + n;
    int *labels = dataLayerOpt.labelsByRecordIndex(recordIndex);
    int labelSize = dataLayerOpt.labelLengthByRecordIndex(recordIndex);

    // inference
    for (int l = 0; l < _numberOfLayers; l++) {
      if (l == 0) { // data layer
        ICI = dataLayerOpt.lengthByRecordIndex(recordIndex);
        in_indices = dataLayerOpt.indicesByRecordIndex(recordIndex);
        in_values = dataLayerOpt.valuesByRecordIndex(recordIndex);
      } else { // prev layer
        ICI = _hiddenlayers[l - 1]->_nodeDataOpt[n].size;
        in_indices = _hiddenlayers[l - 1]->_nodeDataOpt[n].indices;
        in_values = _hiddenlayers[l - 1]->_nodeDataOpt[n].values;
      }
      _hiddenlayers[l]->queryActiveNodeandComputeActivationsOpt(
          in_indices, in_values, ICI,
          l, n, labels, 0, _Sparsity[_numberOfLayers + l], -1);
    }

    // compute softmax
    int noOfClasses = _hiddenlayers[_numberOfLayers - 1]->_nodeDataOpt[n].size;
    float max_act = -222222222;
    int predict_class = -1;
    for (int k = 0; k < noOfClasses; k++) {
      float cur_act = _hiddenlayers[_numberOfLayers - 1]->_nodeDataOpt[n].values[k];
      if (max_act < cur_act) {
        max_act = cur_act;
        predict_class = _hiddenlayers[_numberOfLayers - 1]->_nodeDataOpt[n].indices[k];
      }
    }

    if (std::find(labels, labels + labelSize, predict_class) != labels + labelSize) {
      correctPred++;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

  return correctPred;

}

int Network::predictClass(int **inputIndices, float **inputValues, int *length, int **labels, int *labelsize) {
    alignas(64) int correctPred = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activenodesperlayer[0] = inputIndices[i];
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = length[i];

        //inference
        for (int j = 0; j < _numberOfLayers; j++) {
            _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], 0,
                    _Sparsity[_numberOfLayers+j], -1);
        }

        //compute softmax
        int noOfClasses = sizes[_numberOfLayers];
        float max_act = -222222222;
        int predict_class = -1;
        for (int k = 0; k < noOfClasses; k++) {
            float cur_act = _hiddenlayers[_numberOfLayers - 1]->getNodebyID(activenodesperlayer[_numberOfLayers][k])->getLastActivation(i);
            if (max_act < cur_act) {
                max_act = cur_act;
                predict_class = activenodesperlayer[_numberOfLayers][k];
            }
        }

        if (std::find (labels[i], labels[i]+labelsize[i], predict_class)!= labels[i]+labelsize[i]) {
            correctPred++;
        }

        delete[] sizes;
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activenodesperlayer[j];
            delete[] activeValuesperlayer[j];
        }
        delete[] activenodesperlayer;
        delete[] activeValuesperlayer;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Inference takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

    return correctPred;
}


int Network::ProcessInput(int **inputIndices, float **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash, bool rebuild) {

    float logloss = 0.0;
    int* avg_retrieval = new int[_numberOfLayers]();

    for (int j = 0; j < _numberOfLayers; j++)
        avg_retrieval[j] = 0;


    if(iter%6946==6945 ){
        //_learningRate *= 0.5;
        _hiddenlayers[1]->updateRandomNodes();
    }
    float tmplr = _learningRate;
    if (ADAM) {
        tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
                (1 - pow(BETA1, iter + 1));
    }else{
//        tmplr *= pow(0.9, iter/10.0);
    }

    std::chrono::high_resolution_clock::time_point t1, t2, t3;
    if (DEBUG)
      t1 = std::chrono::high_resolution_clock::now();
    int*** activeNodesPerBatch = new int**[_currentBatchSize];
    float*** activeValuesPerBatch = new float**[_currentBatchSize];
    int** sizesPerBatch = new int*[_currentBatchSize];
#pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        float **activeValuesperlayer = new float *[_numberOfLayers + 1]();
        int *sizes = new int[_numberOfLayers + 1]();

        activeNodesPerBatch[i] = activenodesperlayer;
        activeValuesPerBatch[i] = activeValuesperlayer;
        sizesPerBatch[i] = sizes;

        activenodesperlayer[0] = inputIndices[i];  // inputs parsed from training data file
        activeValuesperlayer[0] = inputValues[i];
        sizes[0] = lengths[i];
        int in;
        //auto t1 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < _numberOfLayers; j++) {
            in = _hiddenlayers[j]->queryActiveNodeandComputeActivations(activenodesperlayer, activeValuesperlayer, sizes, j, i, labels[i], labelsize[i],
                    _Sparsity[j], iter*_currentBatchSize+i);
            avg_retrieval[j] += in;
        }

        //Now backpropagate.
        // layers
        for (int j = _numberOfLayers - 1; j >= 0; j--) {
            Layer* layer = _hiddenlayers[j];
            Layer* prev_layer = _hiddenlayers[j - 1];
            // nodes
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node* node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
                if (j == _numberOfLayers - 1) {
                    //TODO: Compute Extra stats: labels[i];
                    node->ComputeExtaStatsForSoftMax(layer->getNomalizationConstant(i), i, labels[i], labelsize[i]);
                }
                if (j != 0) {
                    node->backPropagate(prev_layer->getAllNodes(), activeNodesPerBatch[i][j], sizesPerBatch[i][j], tmplr, i);
                } else {
                    node->backPropagateFirstLayer(inputIndices[i], inputValues[i], lengths[i], tmplr, i);
                }
            }
        }
    }
    for (int i = 0; i < _currentBatchSize; i++) {
        //Free memory to avoid leaks
        delete[] sizesPerBatch[i];
        for (int j = 1; j < _numberOfLayers + 1; j++) {
            delete[] activeNodesPerBatch[i][j];
            delete[] activeValuesPerBatch[i][j];
        }
        delete[] activeNodesPerBatch[i];
        delete[] activeValuesPerBatch[i];
    }

    delete[] activeNodesPerBatch;
    delete[] activeValuesPerBatch;
    delete[] sizesPerBatch;

    if (DEBUG)
      t2 = std::chrono::high_resolution_clock::now();

    bool tmpRehash;
    bool tmpRebuild;

    for (int l=0; l<_numberOfLayers ;l++) {
        if(rehash & _Sparsity[l]<1){
            tmpRehash=true;
        }else{
            tmpRehash=false;
        }
        if(rebuild & _Sparsity[l]<1){
            tmpRebuild=true;
        }else{
            tmpRebuild=false;
        }
        if (tmpRehash) {
            _hiddenlayers[l]->_hashTables->clear();
        }
        if (tmpRebuild){
            _hiddenlayers[l]->updateTable();
        }
        int ratio = 1;
#pragma omp parallel for
        for (size_t m = 0; m < _hiddenlayers[l]->_noOfNodes; m++)
        {
            Node *tmp = _hiddenlayers[l]->getNodebyID(m);
            int dim = tmp->_dim;
            float* local_weights = new float[dim];
            std::copy(tmp->_weights, tmp->_weights + dim, local_weights);

            if(ADAM){
                for (int d=0; d < dim;d++){
                    float _t = tmp->_t[d];
                    float Mom = tmp->_adamAvgMom[d];
                    float Vel = tmp->_adamAvgVel[d];
                    Mom = BETA1 * Mom + (1 - BETA1) * _t;
                    Vel = BETA2 * Vel + (1 - BETA2) * _t * _t;
                    local_weights[d] += ratio * tmplr * Mom / (sqrt(Vel) + EPS);
                    tmp->_adamAvgMom[d] = Mom;
                    tmp->_adamAvgVel[d] = Vel;
                    tmp->_t[d] = 0;
                }

                tmp->_adamAvgMombias = BETA1 * tmp->_adamAvgMombias + (1 - BETA1) * tmp->_tbias;
                tmp->_adamAvgVelbias = BETA2 * tmp->_adamAvgVelbias + (1 - BETA2) * tmp->_tbias * tmp->_tbias;
                tmp->_bias += ratio*tmplr * tmp->_adamAvgMombias / (sqrt(tmp->_adamAvgVelbias) + EPS);
                tmp->_tbias = 0;
            }
            else
            {
                std::copy(tmp->_mirrorWeights, tmp->_mirrorWeights+(tmp->_dim) , tmp->_weights);
                tmp->_bias = tmp->_mirrorbias;
            }
            if (tmpRehash) {
                int *hashes;
                if(HashFunction==1) {
                    hashes = _hiddenlayers[l]->_wtaHasher->getHash(local_weights);
                }else if (HashFunction==2){
                    hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(local_weights, dim, TOPK);
                }else if (HashFunction==3){
                    hashes = _hiddenlayers[l]->_MinHasher->getHashEasy(_hiddenlayers[l]->_binids, local_weights, dim, TOPK);
                }else if (HashFunction==4){
                    hashes = _hiddenlayers[l]->_srp->getHash(local_weights, dim);
                }

                int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);
                int * bucketIndices = _hiddenlayers[l]->_hashTables->add(hashIndices, m+1);

                delete[] hashes;
                delete[] hashIndices;
                delete[] bucketIndices;
            }

            std::copy(local_weights, local_weights + dim, tmp->_weights);
            delete[] local_weights;
        }
    }

    if (DEBUG)
      t3 = std::chrono::high_resolution_clock::now();
    if (DEBUG && rehash) {
      float timeDiffInMiliseconds1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      std::cout << "Training: FWD/BWD takes " << timeDiffInMiliseconds1/1000 << " milliseconds, ";
      float timeDiffInMiliseconds2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
      std::cout << "ADAM takes " << timeDiffInMiliseconds2/1000 << " milliseconds" << std::endl;

      cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
    }
    return logloss;
}

int Network::ProcessInputOpt(DataLayerOpt &dataLayerOpt, size_t batchIndex,
                             int iter, bool rehash, bool rebuild) {

  float logloss = 0.0;
  int* avg_retrieval = new int[_numberOfLayers]();

  for (int j = 0; j < _numberOfLayers; j++)
    avg_retrieval[j] = 0;


  if(iter%6946==6945 ){
    //_learningRate *= 0.5;
    _hiddenlayers[1]->updateRandomNodes();
  }
  float tmplr = _learningRate;
  if (ADAM) {
    tmplr = _learningRate * sqrt((1 - pow(BETA2, iter + 1))) /
      (1 - pow(BETA1, iter + 1));
  }else{
    //        tmplr *= pow(0.9, iter/10.0);
  }

  std::chrono::high_resolution_clock::time_point t1, t2, t3;
  if (DEBUG)
    t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
  for (int n = 0; n < _currentBatchSize; n++) {
    size_t recordIndex = batchIndex * _currentBatchSize + n;
    int *labels = dataLayerOpt.labelsByRecordIndex(recordIndex);
    int labelSize = dataLayerOpt.labelLengthByRecordIndex(recordIndex);
    int in;

    int ICI;
    int *in_indices;
    float *in_values;
    for (int l = 0; l < _numberOfLayers; l++) {
      if (l == 0) {
        ICI = dataLayerOpt.lengthByRecordIndex(recordIndex);
        in_indices = dataLayerOpt.indicesByRecordIndex(recordIndex);
        in_values = dataLayerOpt.valuesByRecordIndex(recordIndex);
      } else {
        ICI = _hiddenlayers[l - 1]->_nodeDataOpt[n].size;
        in_indices = _hiddenlayers[l - 1]->_nodeDataOpt[n].indices;
        in_values = _hiddenlayers[l - 1]->_nodeDataOpt[n].values;
      }
      in = _hiddenlayers[l]->queryActiveNodeandComputeActivationsOpt(
          in_indices, in_values, ICI, l, n, labels, labelSize,
          _Sparsity[l], iter*_currentBatchSize+n);
      avg_retrieval[l] += in;
    }

    //Now backpropagate.
    // layers
    for (int l = _numberOfLayers - 1; l >= 0; l--) {
      Layer* layer = _hiddenlayers[l];
      Layer* prev_layer = _hiddenlayers[l - 1];
      int OCI = layer->_nodeDataOpt[n].size;

      // ComputeExtaStatsForSoftMaxOpt
      if (l == _numberOfLayers - 1) {
        for (int oci = 0; oci < OCI; oci++) {
          int oc = layer->_nodeDataOpt[n].indices[oci];
          bool &active = layer->_nodeDataOpt[n].active[oci];
          float &value = layer->_nodeDataOpt[n].values[oci];
          float &grad = layer->_nodeDataOpt[n].grads[oci];

          assert(("Input Not Active but still called !! BUG", active));
          value /= layer->getNomalizationConstant(n) + 0.0000001;
          if (find(labels, labels + labelSize, oc)!= labels + labelSize) {
            grad = (1.0/labelSize - value) / _currentBatchSize;
          } else {
            grad = (-value) / _currentBatchSize;
          }
        }
      }

      // backPropagateFirstLayerOpt
      if (l == 0) {
        for (int oci = 0; oci < OCI; oci++) {
          int oc = layer->_nodeDataOpt[n].indices[oci];
          bool &active = layer->_nodeDataOpt[n].active[oci];
          float &value = layer->_nodeDataOpt[n].values[oci];
          float &grad = layer->_nodeDataOpt[n].grads[oci];
          float *prevValues = dataLayerOpt.valuesByRecordIndex(recordIndex);
          float &gb = layer->_biasGrads[oc];

          assert(("Input Not Active but still called !! BUG", active));
          for (int ici = 0; ici < dataLayerOpt.lengthByRecordIndex(recordIndex); ici++) {
            int ic = dataLayerOpt.indicesByRecordIndex(recordIndex)[ici];
            float *gw = &layer->_weightGrads[layer->_previousLayerNumOfNodes * oc];

            float grad_t = grad * prevValues[ici];
            float grad_tsq = grad_t * grad_t;
            if (ADAM) {
              gw[ic] += grad_t;
            } else {
              gw[ic] += tmplr * grad_t;
            }
          }
          if (ADAM) {
            float biasgrad_t = grad;
            float biasgrad_tsq = biasgrad_t * biasgrad_t;
            gb += biasgrad_t;
          } else {
            gb += tmplr * grad;
          }
          active = false;
          grad = 0;
          value = 0;
        }
      } else {
        // backPropagateOpt
        float *prevValues = prev_layer->_nodeDataOpt[n].values;
        float *prevGrads = prev_layer->_nodeDataOpt[n].grads;

        for (int oci = 0; oci < layer->_nodeDataOpt[n].size; oci++) {
          int oc = layer->_nodeDataOpt[n].indices[oci];
          bool &active = layer->_nodeDataOpt[n].active[oci];
          float &value = layer->_nodeDataOpt[n].values[oci];
          float &grad = layer->_nodeDataOpt[n].grads[oci];
          float &gb = layer->_biasGrads[oc];

          assert(("Input Not Active but still called !! BUG", active));
          for (int ici = 0; ici < prev_layer->_nodeDataOpt[n].size; ici++) {
            int ic = prev_layer->_nodeDataOpt[n].indices[ici];
            float *w = &layer->_weights[layer->_previousLayerNumOfNodes * oc];
            float *gw = &layer->_weightGrads[layer->_previousLayerNumOfNodes * oc];
            if (prevValues[ici] > 0) {
              prevGrads[ici] += grad * w[ic];
            }
            float grad_t = grad * prevValues[ici];
            if (ADAM) {
              gw[ic] += grad_t;
            } else {
              gw[ic] += tmplr * grad_t;
            }
          }
          if (ADAM) {
            float biasgrad_t = grad;
            float biasgrad_tsq = biasgrad_t * biasgrad_t;
            gb += biasgrad_t;
          } else {
            gb += tmplr * grad;
          }
          active = false;
          grad = 0;
          value = 0;
        }
      }
    }
  }

  if (DEBUG)
    t2 = std::chrono::high_resolution_clock::now();

  bool tmpRehash;
  bool tmpRebuild;

  for (int l=0; l<_numberOfLayers ;l++) {
    if(rehash & _Sparsity[l]<1){
      tmpRehash=true;
    }else{
      tmpRehash=false;
    }
    if(rebuild & _Sparsity[l]<1){
      tmpRebuild=true;
    }else{
      tmpRebuild=false;
    }
    if (tmpRehash) {
      _hiddenlayers[l]->_hashTables->clear();
    }
    if (tmpRebuild){
      _hiddenlayers[l]->updateTable();
    }
    int ratio = 1;
    Layer *layer = _hiddenlayers[l];
    size_t OC = _hiddenlayers[l]->_noOfNodes;
    size_t IC = _hiddenlayers[l]->_previousLayerNumOfNodes;
#pragma omp parallel for
    for (size_t oc = 0; oc < OC; oc++) {
      if (ADAM) {
        for (size_t ic = 0; ic < IC; ic++) {
          float &gw = layer->_weightGrads[IC * oc + ic];
          float &mom = layer->_adamAvgMom[IC * oc + ic];
          float &vel = layer->_adamAvgVel[IC * oc + ic];
          float &w = layer->_weights[IC * oc + ic];
          mom = BETA1 * mom + (1 - BETA1) * gw;
          vel = BETA2 * vel + (1 - BETA2) * gw * gw;
          w += ratio * tmplr * mom / (sqrt(vel) + EPS);
          gw = 0;
        }
        float &gb = layer->_biasGrads[oc];
        float &b = layer->_bias[oc];
        float &bmom = layer->_adamAvgMomBias[oc];
        float &bvel = layer->_adamAvgVelBias[oc];
        bmom = BETA1 * bmom + (1 - BETA1) * gb;
        bvel = BETA2 * bvel + (1 - BETA2) * gb * gb;
        b += ratio * tmplr * bmom / (sqrt(bvel) + EPS);
        gb = 0;
      }

      if (tmpRehash) {
        size_t dim = IC;
        float *local_weights = &layer->_weights[IC * oc];
        int *hashes;
        if(HashFunction==1) {
          hashes = _hiddenlayers[l]->_wtaHasher->getHash(local_weights);
        }else if (HashFunction==2){
          hashes = _hiddenlayers[l]->_dwtaHasher->getHashEasy(local_weights, dim, TOPK);
        }else if (HashFunction==3){
          hashes = _hiddenlayers[l]->_MinHasher->getHashEasy(_hiddenlayers[l]->_binids, local_weights, dim, TOPK);
        }else if (HashFunction==4){
          hashes = _hiddenlayers[l]->_srp->getHash(local_weights, dim);
        }

        int *hashIndices = _hiddenlayers[l]->_hashTables->hashesToIndex(hashes);
        int * bucketIndices = _hiddenlayers[l]->_hashTables->add(hashIndices, oc+1);

        delete[] hashes;
        delete[] hashIndices;
        delete[] bucketIndices;
      }
    }
  }

  if (DEBUG)
    t3 = std::chrono::high_resolution_clock::now();
  if (DEBUG && rehash) {
    float timeDiffInMiliseconds1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Training: FWD/BWD takes " << timeDiffInMiliseconds1/1000 << " milliseconds, ";
    float timeDiffInMiliseconds2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    std::cout << "ADAM takes " << timeDiffInMiliseconds2/1000 << " milliseconds" << std::endl;

    cout << "Avg sample size = " << avg_retrieval[0]*1.0/_currentBatchSize<<" "<<avg_retrieval[1]*1.0/_currentBatchSize << endl;
  }
  return logloss;
}



void Network::saveWeights(string file)
{
    for (int i=0; i< _numberOfLayers; i++){
        _hiddenlayers[i]->saveWeights(file);
    }
}


Network::~Network() {

    delete[] _sizesOfLayers;
    for (int i=0; i< _numberOfLayers; i++){
        delete _hiddenlayers[i];
    }
    delete[] _hiddenlayers;
    delete[] _layersTypes;
}
