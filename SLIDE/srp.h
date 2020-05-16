#include <vector>
#pragma once
using namespace std;

template <class T>
class SparseRandomProjection 
{
private:
	size_t _dim;
	size_t _numhashes, _samSize;
	short ** _randBits;
	int ** _indices;
public:
	SparseRandomProjection(size_t dimention, size_t numOfHashes, int ratio);
	int * getHash(T * vector, int length);
	int * getHashSparse(int* indices, T *values, size_t length);
	~SparseRandomProjection();
};
