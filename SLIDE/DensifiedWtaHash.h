#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <climits>
#include <iostream>
#include <random>
#include <vector>
#include <string.h>
#include "MurmurHash.h"
/*
*  Algorithm from the paper Densified Winner Take All (WTA) Hashing for Sparse Datasets. Beidi Chen, Anshumali Shrivastava
*/
using namespace std;

class DensifiedWtaHash
{
private:
    int *_randHash, _randa, _numhashes, _rangePow,_lognumhash, *_indices, *_pos, _permute;
public:
    DensifiedWtaHash(int numHashes, int noOfBitsToHash);
    template <class T>
    int * getHash(int* indices, T* data, int dataLen);
    int getRandDoubleHash(int binid, int count);
    template <class T>
    int * getHashEasy(T* data, int dataLen, int topK, int stride = 1);
    ~DensifiedWtaHash();
};
