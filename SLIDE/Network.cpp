#include "Network.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include "Config.h"
#include <omp.h>
#define DEBUG 1
using namespace std;


template <class T, class Tp>
Network<T, Tp>::Network(int *sizesOfLayers, NodeType *layersTypes, int noOfLayers, int batchSize, float lr, int inputdim,  int* K, int* L, int* RangePow, float* Sparsity, cnpy::npz_t arr) {

    _numberOfLayers = noOfLayers;
    _hiddenlayers = new Layer<T, Tp> *[noOfLayers];
    _sizesOfLayers = sizesOfLayers;
    _layersTypes = layersTypes;
    _learningRate = lr;
    _currentBatchSize = batchSize;
    _Sparsity = Sparsity;


    for (int i = 0; i < noOfLayers; i++) {
        if (i != 0) {
            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            Tp* weight, *bias;
            float *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<Tp>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<Tp>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }
            _hiddenlayers[i] = new Layer<T, Tp>(sizesOfLayers[i], sizesOfLayers[i - 1], i, _layersTypes[i], _currentBatchSize,  K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        } else {

            cnpy::NpyArray weightArr, biasArr, adamArr, adamvArr;
            Tp* weight, *bias;
            float *adamAvgMom, *adamAvgVel;
            if(LOADWEIGHT){
                weightArr = arr["w_layer_"+to_string(i)];
                weight = weightArr.data<Tp>();
                biasArr = arr["b_layer_"+to_string(i)];
                bias = biasArr.data<Tp>();

                adamArr = arr["am_layer_"+to_string(i)];
                adamAvgMom = adamArr.data<float>();
                adamvArr = arr["av_layer_"+to_string(i)];
                adamAvgVel = adamvArr.data<float>();
            }
            _hiddenlayers[i] = new Layer<T, Tp>(sizesOfLayers[i], inputdim, i, _layersTypes[i], _currentBatchSize, K[i], L[i], RangePow[i], Sparsity[i], weight, bias, adamAvgMom, adamAvgVel);
        }
    }
    cout << "after layer" << endl;
}


template <class T, class Tp>
Layer<T, Tp> *Network<T, Tp>::getLayer(int LayerID) {
    if (LayerID < _numberOfLayers)
        return _hiddenlayers[LayerID];
    else {
        cout << "LayerID out of bounds" << endl;
        //TODO:Handle
    }
}

template <class T, class Tp>
int Network<T, Tp>::predictClassOpt(DataLayerOpt<T> &dataLayerOpt, size_t batchIndex) {
  alignas(64) int correctPred = 0;

  auto t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for reduction(+:correctPred)
  for (int n = 0; n < _currentBatchSize; n++) {
    int ICI;
    int *in_indices;
    T *in_values;
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
    int max_act_index = 0;
    for (int k = 0; k < noOfClasses; k++) {
      float cur_act = _hiddenlayers[_numberOfLayers - 1]->_nodeDataOpt[n].values[k];
      if (max_act < cur_act) {
        max_act = cur_act;
        max_act_index = k;
      }
    }
    predict_class = _hiddenlayers[_numberOfLayers - 1]->_nodeDataOpt[n].indices[max_act_index];

    if (std::find(labels, labels + labelSize, predict_class) != labels + labelSize) {
      correctPred++;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  float timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Inference takes " << timeDiffInMiliseconds/1000 << " milliseconds" << std::endl;

  return correctPred;

}

template <class T, class Tp>
int Network<T, Tp>::predictClass(int **inputIndices, T **inputValues, int *length, int **labels, int *labelsize) {
    alignas(64) int correctPred = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:correctPred)
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        T **activeValuesperlayer = new T *[_numberOfLayers + 1]();
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


template <class T, class Tp>
int Network<T, Tp>::ProcessInput(int **inputIndices, T **inputValues, int *lengths, int **labels, int *labelsize, int iter, bool rehash, bool rebuild) {

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
    T*** activeValuesPerBatch = new T**[_currentBatchSize];
    int** sizesPerBatch = new int*[_currentBatchSize];
#pragma omp parallel for
    for (int i = 0; i < _currentBatchSize; i++) {
        int **activenodesperlayer = new int *[_numberOfLayers + 1]();
        T **activeValuesperlayer = new T *[_numberOfLayers + 1]();
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
            Layer<T, Tp>* layer = _hiddenlayers[j];
            Layer<T, Tp>* prev_layer = _hiddenlayers[j - 1];
            // nodes
            for (int k = 0; k < sizesPerBatch[i][j + 1]; k++) {
                Node<T, Tp>* node = layer->getNodebyID(activeNodesPerBatch[i][j + 1][k]);
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
            Node<T, Tp> *tmp = _hiddenlayers[l]->getNodebyID(m);
            int dim = tmp->_dim;
            float* local_weights = new float[dim];
            std::copy(tmp->_weights, tmp->_weights + dim, local_weights);

            if(ADAM){
                for (int d=0; d < dim;d++){
                    T _t = tmp->_t[d];
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

template <class T, class Tp>
int Network<T, Tp>::ProcessInputOpt(DataLayerOpt<T> &dataLayerOpt, size_t batchIndex,
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
    T *in_values;
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

    // Now backpropagate.
    for (int l = _numberOfLayers - 1; l >= 0; l--) {
      Layer<T, Tp>* layer = _hiddenlayers[l];
      Layer<T, Tp>* prev_layer = _hiddenlayers[l - 1];

      if (l == _numberOfLayers - 1)
        layer->computeExtraStatsForSoftMaxOpt(labels, labelSize, n,
                                              _currentBatchSize);
      if (l == 0)
        layer->backPropagateFirstLayerOpt(dataLayerOpt, n, recordIndex, tmplr);
      else 
        layer->backPropagateOpt(prev_layer, n, tmplr);
    }
  }

  if (DEBUG)
    t2 = std::chrono::high_resolution_clock::now();

  bool tmpRehash;
  bool tmpRebuild;
  for (int l = 0; l < _numberOfLayers; l++) {
    if (rehash && _Sparsity[l] < 1){
      tmpRehash = true;
    } else {
      tmpRehash = false;
    }
    if (rebuild && _Sparsity[l] < 1){
      tmpRebuild = true;
    } else {
      tmpRebuild = false;
    }
    if (tmpRehash) {
      _hiddenlayers[l]->_hashTables->clear();
    }
    if (tmpRebuild){
      _hiddenlayers[l]->updateTable();
    }
    int ratio = 1;
    Layer<T, Tp> *layer = _hiddenlayers[l];
    size_t OC = _hiddenlayers[l]->_noOfNodes;
    size_t IC = _hiddenlayers[l]->_previousLayerNumOfNodes;
    bool isOIWeights = _hiddenlayers[l]->_weightsOrder == WeightsOrder::OI;
    if (ADAM) {
#if OPT_IA && OPT_VEC512
      constexpr int V = 16;
      int I2 = (IC + V - 1) / V;
      int O2 = (OC + V - 1) / V;

      __m512 vec_one = _mm512_set1_ps(1.0f);
      __m512 vec_zero = _mm512_setzero_ps();
      __m512 vec_BETA1 = _mm512_set1_ps(BETA1);
      __m512 vec_BETA2 = _mm512_set1_ps(BETA2);
      __m512 vec_ratio = _mm512_set1_ps(ratio);
      __m512 vec_tmplr = _mm512_set1_ps(tmplr);
      __m512 vec_EPS = _mm512_set1_ps(EPS);

      auto vecAdamWeights = [&](int Vx, int idx) {
        __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
        __m512 vec_mom, vec_vel, vec_w, vec_gw;

        vec_mom = _mm512_maskz_load<float>(k, &layer->_adamAvgMom[idx]);
        vec_vel = _mm512_maskz_load<float>(k, &layer->_adamAvgVel[idx]);
        if (std::is_same<Tp, float>::value) {
          vec_w = _mm512_maskz_load_ps(k, &layer->_weights[idx]);
        } else {
          __m256i vec_w0 = _mm256_maskz_loadu_epi16(k, &layer->_weightsLo[idx]);
          __m256i vec_w1 = _mm256_maskz_loadu_epi16(k, &layer->_weights[idx]);
          __m512i vec_ww0 = _mm512_cvtepu16_epi32(vec_w0);
          __m512i vec_ww1 = _mm512_cvtepu16_epi32(vec_w1);
          vec_w = _mm512_castsi512_ps(
              _mm512_or_epi32(vec_ww0, _mm512_bslli_epi128(vec_ww1, 2)));
        }
        vec_gw = _mm512_maskz_load<T>(k, &layer->_weightGrads[idx]);

        vec_mom = vec_BETA1 * vec_mom + (vec_one - vec_BETA1) * vec_gw;
        vec_vel = vec_BETA2 * vec_vel + (vec_one - vec_BETA2) * vec_gw * vec_gw;
        vec_w += vec_ratio * vec_tmplr * vec_mom / (_mm512_sqrt_ps(vec_vel) + vec_EPS);

        _mm512_mask_store<float>(&layer->_adamAvgMom[idx], k, vec_mom);
        _mm512_mask_store<float>(&layer->_adamAvgVel[idx], k, vec_vel);
        if (std::is_same<Tp, float>::value) {
          _mm512_mask_store_ps(&layer->_weights[idx], k, vec_w);
        } else {
          _mm256_mask_storeu_epi16(&layer->_weightsLo[idx], k,
                                   _mm512_cvtepi32_epi16(_mm512_castps_si512(vec_w)));
          _mm256_mask_storeu_epi16(&layer->_weights[idx], k,
                                   _mm512_cvtepi32_epi16(
                                       _mm512_bsrli_epi128(_mm512_castps_si512(vec_w), 2)));
        }
        _mm512_mask_store<T>(&layer->_weightGrads[idx], k, vec_zero);
      };
      auto vecAdamBias = [&](int Vx, int idx) {
        __mmask16 k = _cvtu32_mask16((1 << Vx) - 1);
        __m512 vec_mom, vec_vel, vec_b, vec_gb;

        vec_mom = _mm512_maskz_load<float>(k, &layer->_adamAvgMomBias[idx]);
        vec_vel = _mm512_maskz_load<float>(k, &layer->_adamAvgVelBias[idx]);
        vec_b = _mm512_maskz_load<Tp>(k, &layer->_bias[idx]);
        vec_gb = _mm512_maskz_load<T>(k, &layer->_biasGrads[idx]);

        vec_mom = vec_BETA1 * vec_mom + (vec_one - vec_BETA1) * vec_gb;
        vec_vel = vec_BETA2 * vec_vel + (vec_one - vec_BETA2) * vec_gb * vec_gb;
        vec_b += vec_ratio * vec_tmplr * vec_mom / (_mm512_sqrt_ps(vec_vel) + vec_EPS);

        _mm512_mask_store<float>(&layer->_adamAvgMomBias[idx], k, vec_mom);
        _mm512_mask_store<float>(&layer->_adamAvgVelBias[idx], k, vec_vel);
        _mm512_mask_store<Tp>(&layer->_bias[idx], k, vec_b);
        _mm512_mask_store<T>(&layer->_biasGrads[idx], k, vec_zero);
      };
#else
      auto adamWeights = [&](int idx) {
        float w;
        if (std::is_same<Tp, float>::value) {
          w = layer->_weights[idx];
        } else {
          float_raw f;
          f.wraw[1] = layer->_weights[idx];
          f.wraw[0] = layer->_weightsLo[idx];
          w = f.fraw;
        }
        T &gw = layer->_weightGrads[idx];
        float &mom = layer->_adamAvgMom[idx];
        float &vel = layer->_adamAvgVel[idx];

        mom = BETA1 * mom + (1 - BETA1) * gw;
        vel = BETA2 * vel + (1 - BETA2) * gw * gw;
        w += ratio * tmplr * mom / (sqrt(vel) + EPS);
        gw = 0;
        if (std::is_same<Tp, float>::value) {
          layer->_weights[idx] = w;
        } else {
          float_raw f;
          f.fraw = w;
          layer->_weights[idx] = f.wraw[1];
          layer->_weightsLo[idx] = f.wraw[0];
        }
      };

      auto adamBias = [&](int oc) {
        T &gb = layer->_biasGrads[oc];
        Tp &b = layer->_bias[oc];
        float &bmom = layer->_adamAvgMomBias[oc];
        float &bvel = layer->_adamAvgVelBias[oc];
        bmom = BETA1 * bmom + (1 - BETA1) * gb;
        bvel = BETA2 * bvel + (1 - BETA2) * gb * gb;
        b += ratio * tmplr * bmom / (sqrt(bvel) + EPS);
        gb = 0;
      };
#endif

#if OPT_IA && OPT_VEC512
      if (isOIWeights) {
        #pragma omp parallel for
        for (size_t oc = 0; oc < OC; oc++) {
          int Vr = IC % V ? IC % V : V;
          for (int i2 = 0; i2 < I2; i2++) {
            int Vx = i2 == I2 - 1 ? Vr : V;
            int idx = IC * oc + i2 * V;
            vecAdamWeights(Vx, idx);
          }
          if (tmpRehash)
            layer->addtoHashTable(&layer->_weights[IC * oc], IC, 0, oc, 1);
        }
      } else { // IO
        #pragma omp parallel for
        for (size_t ic = 0; ic < IC; ic++) {
          int Vr = OC % V ? OC % V : V;
          for (int o2 = 0; o2 < O2; o2++) {
            int Vx = o2 == O2 - 1 ? Vr : V;
            int idx = OC * ic + o2 * V;
            vecAdamWeights(Vx, idx);
          }
        }
        if (tmpRehash) {
          #pragma omp parallel for
          for (size_t oc = 0; oc < OC; oc++)
            layer->addtoHashTable(&layer->_weights[oc], IC, 0, oc, OC);
        }
      }
      #pragma omp parallel for
      for (size_t o2 = 0; o2 < O2; o2++) {
        int Vr = OC % V ? OC % V : V;
        int Vx = o2 == O2 - 1 ? Vr : V;
        int idx = o2 * V;
        vecAdamBias(Vx, idx);
      }

#else
      if (isOIWeights) {
        #pragma omp parallel for
        for (size_t oc = 0; oc < OC; oc++) {
          for (size_t ic = 0; ic < IC; ic++) {
            int idx = IC * oc + ic;
            adamWeights(idx);
          }
          adamBias(oc);
          if (tmpRehash)
            layer->addtoHashTable(&layer->_weights[IC * oc], IC, 0, oc, 1);
        }
      } else {
        #pragma omp parallel for
        for (size_t ic = 0; ic < IC; ic++) {
          for (size_t oc = 0; oc < OC; oc++) {
            int idx = ic * OC + oc;
            adamWeights(idx);
          }
        }
        #pragma omp parallel for
        for (size_t oc = 0; oc < OC; oc++) {
          adamBias(oc);
          if (tmpRehash)
            layer->addtoHashTable(&layer->_weights[oc], IC, 0, oc, OC);
        }
      }
#endif
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



template <class T, class Tp>
void Network<T, Tp>::saveWeights(string file)
{
    for (int i=0; i< _numberOfLayers; i++){
        _hiddenlayers[i]->saveWeights(file);
    }
}


template <class T, class Tp>
Network<T, Tp>::~Network() {

    delete[] _sizesOfLayers;
    for (int i=0; i< _numberOfLayers; i++){
        delete _hiddenlayers[i];
    }
    delete[] _hiddenlayers;
    delete[] _layersTypes;
}

template class Network<float, float>;
template class Network<bfloat16, float>;
template class Network<bfloat16, bfloat16>;
