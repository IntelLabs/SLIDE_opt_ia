#include "DensifiedMinhash.h"
#include <random>
#include <iostream>
#include <math.h>
#include <vector>
#include <climits>
#include <algorithm>
#include <queue>
#include "Bfloat16.h"
using namespace std;

template <class T> using PAIR = pair<int, T>;

template <class T>
struct cmp {
    bool operator()(const PAIR<T> &a, const PAIR<T> &b) {
        return a.second > b.second; //lower is better
    };
};

template <class T>
DensifiedMinhash<T>::DensifiedMinhash(int numHashes, int noOfBitsToHash)
{

    _numhashes = numHashes;
    _rangePow = noOfBitsToHash;
    _lognumhash = log2(numHashes);


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, INT_MAX);

    _randa = dis(gen);
    if (_randa % 2 == 0)
        _randa++;
    _randHash = new int[2];
    _randHash[0] = dis(gen);
    if (_randHash[0] % 2 == 0)
        _randHash[0]++;
    _randHash[1] = dis(gen);
    if (_randHash[1] % 2 == 0)
        _randHash[1]++;

}

template <class T>
void DensifiedMinhash<T>::getMap(int n, int* binids)
{
    int range = 1 << _rangePow;
    // binsize is the number of times the range is larger than the total number of hashes we need.
    int binsize = ceil(1.0*range / _numhashes);
    for (int i = 0; i < n; i++)
    {
        unsigned int h = i;
        h *= _randa;
        h ^= h >> 13;
        h *= 0x85ebca6b;
//        unsigned int curhash = (unsigned int)(((unsigned int)h*i) << 5);
        uint32_t curhash = MurmurHash ((char *)&i, (uint32_t) sizeof(i), (uint32_t)_randa);
        curhash = curhash & ((1<<_rangePow)-1);
        binids[i] = (int)floor(curhash / binsize);;
    }

}


template <class T>
int * DensifiedMinhash<T>::getHashEasy(int* binids, T* data, int dataLen, int topK)
{

    // binsize is the number of times the range is larger than the total number of hashes we need.
// read the data and add it to priority queue O(dlogk approx 7d) with index as key and values as priority value, get topk index O(1) and apply minhash on retuned index.

    priority_queue<PAIR<T>, vector<PAIR<T> >, cmp<T>> pq;

    for (int i = 0; i < topK; i++)
    {
        pq.push(std::make_pair(i,data[i]));
    }

    for (int i = topK; i < dataLen; i++)
    {
        pq.push(std::make_pair(i,data[i]));
        pq.pop();
    }



    int *hashes = new int[_numhashes];
    //float *values = new float[_numhashes];
    int *hashArray = new int[_numhashes];

    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
    }


    for (int i = 0; i < topK; i++)
    {
        PAIR<T> pair = pq.top();
        pq.pop();
        int index = pair.first;
        int binid = binids[index];
        if (hashes[binid] < index) {
            hashes[binid] = index;
        }
    }


    for (int i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    return hashArray;
}

template <class T>
int * DensifiedMinhash<T>::getHash(int* indices, T* data, int* binids, int dataLen)
{
    int *hashes = new int[_numhashes];
    int *hashArray = new int[_numhashes];

    for (int i = 0; i < _numhashes; i++)
    {
        hashes[i] = INT_MIN;
    }

    if (dataLen<0){

    }

    for (int i = 0; i < dataLen; i++)
    {
        int binid = binids[indices[i]];

        if (hashes[binid] < indices[i]){
            hashes[binid] = indices[i];
        }
    }

    for (int i = 0; i < _numhashes; i++)
    {
        int next = hashes[i];
        if (next != INT_MIN)
        {
            hashArray[i] = hashes[i];
            continue;
        }
        int count = 0;
        while (next == INT_MIN)
        {
            count++;
            int index = std::min(
                    getRandDoubleHash(i, count),
                    _numhashes);

            next = hashes[index]; // Kills GPU.
            if (count > 100) // Densification failure.
                break;
        }
        hashArray[i] = next;
    }
    delete[] hashes;
    //   delete[] values;
    return hashArray;
}


template <class T>
int DensifiedMinhash<T>::getRandDoubleHash(int binid, int count) {
    unsigned int tohash = ((binid + 1) << 6) + count;
    return (_randHash[0] * tohash << 3) >> (32 - _lognumhash); // _lognumhash needs to be ceiled.
}


template <class T>
DensifiedMinhash<T>::~DensifiedMinhash()
{
    delete[] _randHash;
}

template class DensifiedMinhash<float>;
template class DensifiedMinhash<bfloat16>;
