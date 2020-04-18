#include "Node.h"
#include <random>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <sys/mman.h>
#include "Config.h"
#include <x86intrin.h>
#include "Bfloat16.h"

using namespace std;

template <class T>
Node<T>::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, T *weights, T bias, float *adamAvgMom, float *adamAvgVel)
{
	_dim = dim;
	_IDinLayer = nodeID;
	_type = type;
	_layerNum = layerID;
    _currentBatchsize = batchsize;

	if (ADAM)
	{
		_adamAvgMom = adamAvgMom;
		_adamAvgVel = adamAvgVel;
		_t = new T[_dim]();

	}

	_train = new train<T>[_currentBatchsize];
	_activeInputs = 0;

    _weights = weights;
    _bias = bias;
	_mirrorbias = _bias;

}

template <class T>
void Node<T>::Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, T *weights, T bias, float *adamAvgMom, float *adamAvgVel, train<T>* train_blob)
{
    _dim = dim;
    _IDinLayer = nodeID;
    _type = type;
    _layerNum = layerID;
    _currentBatchsize = batchsize;

    if (ADAM)
    {
        _adamAvgMom = adamAvgMom;
        _adamAvgVel = adamAvgVel;
        _t = new T[_dim]();

    }

    _train = train_blob + nodeID * batchsize;
    _activeInputs = 0;

    _weights = weights;
    _bias = bias;
    _mirrorbias = _bias;

}

template <class T>
T Node<T>::getLastActivation(int inputID)
{
	if(_train[inputID]._ActiveinputIds != 1)
		return 0.0;
	return _train[inputID]._lastActivations;
}


template <class T>
void Node<T>::incrementDelta(int inputID, T incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	if (_train[inputID]._lastActivations > 0)
	    _train[inputID]._lastDeltaforBPs += incrementValue;
}

template <class T>
bool Node<T>::getInputActive(int inputID)
{
    return _train[inputID]._ActiveinputIds == 1;
}

template <class T>
bool Node<T>::getActiveInputs(void)
{
    return _activeInputs > 0;
}

template <class T>
T Node<T>::getActivation(int* indices, T* values, int length, int inputID)
{
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
	if (_train[inputID]._ActiveinputIds != 1) {
	    _train[inputID]._ActiveinputIds = 1; //activate input
	    _activeInputs++;
	}

#define PF_STR 32

  float res = _bias;
	for (int i = 0; i < length; i++)
	{
	    res += _weights[indices[i]] * values[i];
      if (i + PF_STR < length)
      _mm_prefetch(&_weights[indices[i + PF_STR]], _MM_HINT_T0);
	}

	switch (_type)
	{
	case NodeType::ReLU: // TODO
		if (res < 0) {
        res = 0;
		    _train[inputID]._lastGradients = 1;
		    _train[inputID]._lastDeltaforBPs = 0;

        }else{
            _train[inputID]._lastGradients = 0;
		}
		break;
	case NodeType::Softmax:

		break;
	default:
		cout << "Invalid Node type from Constructor" <<endl;
		break;
	}

  return res;
}


template <class T>
void Node<T>::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds ==1));

	_train[inputID]._lastActivations /= normalizationConstant + 0.0000001;

	//TODO:check  gradient
	_train[inputID]._lastGradients = 1;
	if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
	    _train[inputID]._lastDeltaforBPs = (1.0/labelsize - _train[inputID]._lastActivations) / _currentBatchsize;
	}
	else {
	    _train[inputID]._lastDeltaforBPs = (-_train[inputID]._lastActivations) / _currentBatchsize;
	}
}

template <class T>
void Node<T>::ComputeExtaStatsForSoftMaxOpt(T &value, T &grad, float normalizationConstant, int inputID, int* label, int labelsize)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds ==1));

	value /= normalizationConstant + 0.0000001;

	//TODO:check  gradient
	if (find (label, label+labelsize, _IDinLayer)!= label+labelsize) {
	    grad = (1.0/labelsize - value) / _currentBatchsize;
	}
	else {
	    grad = (-value) / _currentBatchsize;
	}
}

template <class T>
void Node<T>::backPropagate(Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
	    Node* prev_node = &(previousNodes[previousLayerActiveNodeIds[i]]);
	    prev_node->incrementDelta(inputID, _train[inputID]._lastDeltaforBPs * _weights[previousLayerActiveNodeIds[i]]);

		T grad_t = _train[inputID]._lastDeltaforBPs * prev_node->getLastActivation(inputID);

		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		T biasgrad_t = _train[inputID]._lastDeltaforBPs;
		T biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
    {
        _mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
    }

	_train[inputID]._ActiveinputIds = 0;
	_train[inputID]._lastDeltaforBPs = 0;
	_train[inputID]._lastActivations = 0;
	_activeInputs--;

}

template <class T>
void Node<T>::backPropagateOpt(T &value, T &grad, T *prevValues, T *prevGrads, Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < previousLayerActiveNodeSize; i++)
	{
		//UpdateDelta before updating weights
	    Node* prev_node = &(previousNodes[previousLayerActiveNodeIds[i]]);
      if (prevValues[i] > 0) {
        prevGrads[i] += grad * _weights[previousLayerActiveNodeIds[i]];
      }


		T grad_t = grad * prevValues[i];

		if (ADAM)
		{
			_t[previousLayerActiveNodeIds[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[previousLayerActiveNodeIds[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		T biasgrad_t = grad;
		T biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
    {
        _mirrorbias += learningRate * grad;
    }

	_train[inputID]._ActiveinputIds = 0;
	grad = 0;
  value = 0;
	_activeInputs--;

}


template <class T>
void Node<T>::backPropagateFirstLayer(int* nnzindices, T* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < nnzSize; i++)
	{
		T grad_t = _train[inputID]._lastDeltaforBPs * nnzvalues[i];
		T grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		T biasgrad_t = _train[inputID]._lastDeltaforBPs;
		T biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
	{
		_mirrorbias += learningRate * _train[inputID]._lastDeltaforBPs;
	}

	_train[inputID]._ActiveinputIds = 0;//deactivate inputIDs
	_train[inputID]._lastDeltaforBPs = 0;
	_train[inputID]._lastActivations = 0;
    _activeInputs--;
}

template <class T>
void Node<T>::backPropagateFirstLayerOpt(T &value, T &grad, int* nnzindices, T* nnzvalues, int nnzSize, float learningRate, int inputID)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	for (int i = 0; i < nnzSize; i++)
	{
		T grad_t = grad * nnzvalues[i];
		T grad_tsq = grad_t * grad_t;
		if (ADAM)
		{
			_t[nnzindices[i]] += grad_t;
		}
		else
		{
			_mirrorWeights[nnzindices[i]] += learningRate * grad_t;
		}
	}

	if (ADAM)
	{
		T biasgrad_t = grad;
		T biasgrad_tsq = biasgrad_t * biasgrad_t;
		_tbias += biasgrad_t;
	}
	else
	{
		_mirrorbias += learningRate * grad;
	}

	_train[inputID]._ActiveinputIds = 0;//deactivate inputIDs
	grad = 0;
  value = 0;
    _activeInputs--;
}


template <class T>
void Node<T>::SetlastActivation(int inputID, T realActivation)
{
    _train[inputID]._lastActivations = realActivation;
}

template <class T>
Node<T>::~Node()
{

	delete[] _indicesInTables;
	delete[] _indicesInBuckets;

	if (ADAM)
	{
		delete[] _adamAvgMom;
		delete[] _adamAvgVel;
		delete[] _t;
	}
}


// for debugging gradients.
template <class T>
T Node<T>::purturbWeight(int weightid, T delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}


template <class T>
T Node<T>::getGradient(int weightid, int inputID, T InputVal)
{
	return -_train[inputID]._lastDeltaforBPs * InputVal;
}

template class Node<float>;
template class Node<bfloat16>;
