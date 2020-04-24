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

template <class T, class Tp>
Node<T, Tp>::Node(int dim, int nodeID, int layerID, NodeType type, int batchsize, Tp *weights, Tp bias, float *adamAvgMom, float *adamAvgVel)
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
		_t = new float[_dim]();

	}

	_train = new train<T>[_currentBatchsize];
	_activeInputs = 0;

    _weights = weights;
    _bias = bias;
	_mirrorbias = _bias;

}

template <class T, class Tp>
void Node<T, Tp>::Update(int dim, int nodeID, int layerID, NodeType type, int batchsize, Tp *weights, Tp bias, float *adamAvgMom, float *adamAvgVel, train<T>* train_blob)
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
        _t = new float[_dim]();

    }

    _train = train_blob + nodeID * batchsize;
    _activeInputs = 0;

    _weights = weights;
    _bias = bias;
    _mirrorbias = _bias;

}

template <class T, class Tp>
T Node<T, Tp>::getLastActivation(int inputID)
{
	if(_train[inputID]._ActiveinputIds != 1)
		return 0.0;
	return _train[inputID]._lastActivations;
}


template <class T, class Tp>
void Node<T, Tp>::incrementDelta(int inputID, T incrementValue)
{
	assert(("Input Not Active but still called !! BUG", _train[inputID]._ActiveinputIds == 1));
	if (_train[inputID]._lastActivations > 0)
	    _train[inputID]._lastDeltaforBPs += incrementValue;
}

template <class T, class Tp>
bool Node<T, Tp>::getInputActive(int inputID)
{
    return _train[inputID]._ActiveinputIds == 1;
}

template <class T, class Tp>
bool Node<T, Tp>::getActiveInputs(void)
{
    return _activeInputs > 0;
}

template <class T, class Tp>
T Node<T, Tp>::getActivation(int* indices, T* values, int length, int inputID)
{
	assert(("Input ID more than Batch Size", inputID <= _currentBatchsize));

	//FUTURE TODO: shrink batchsize and check if input is alread active then ignore and ensure backpopagation is ignored too.
	if (_train[inputID]._ActiveinputIds != 1) {
	    _train[inputID]._ActiveinputIds = 1; //activate input
	    _activeInputs++;
	}

	_train[inputID]._lastActivations = 0;
	for (int i = 0; i < length; i++)
	{
	    _train[inputID]._lastActivations += _weights[indices[i]] * values[i];
	}
	_train[inputID]._lastActivations += _bias;

	switch (_type)
	{
	case NodeType::ReLU:
		if (_train[inputID]._lastActivations < 0) {
		    _train[inputID]._lastActivations = 0;
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

	return _train[inputID]._lastActivations;
}


template <class T, class Tp>
void Node<T, Tp>::ComputeExtaStatsForSoftMax(float normalizationConstant, int inputID, int* label, int labelsize)
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


template <class T, class Tp>
void Node<T, Tp>::backPropagate(Node* previousNodes, int* previousLayerActiveNodeIds, int previousLayerActiveNodeSize, float learningRate, int inputID)
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


template <class T, class Tp>
void Node<T, Tp>::backPropagateFirstLayer(int* nnzindices, T* nnzvalues, int nnzSize, float learningRate, int inputID)
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

template <class T, class Tp>
void Node<T, Tp>::SetlastActivation(int inputID, T realActivation)
{
    _train[inputID]._lastActivations = realActivation;
}

template <class T, class Tp>
Node<T, Tp>::~Node()
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
template <class T, class Tp>
T Node<T, Tp>::purturbWeight(int weightid, T delta)
{
	_weights[weightid] += delta;
	return _weights[weightid];
}


template <class T, class Tp>
T Node<T, Tp>::getGradient(int weightid, int inputID, T InputVal)
{
	return -_train[inputID]._lastDeltaforBPs * InputVal;
}

template class Node<float, float>;
template class Node<bfloat16, float>;
template class Node<bfloat16, bfloat16>;
