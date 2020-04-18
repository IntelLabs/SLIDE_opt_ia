#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include "Bfloat16.h"


template <class T>
struct DataLayerOpt {
  size_t numRecords_;
  size_t numFeatures_;
  size_t numLabels_;

  // features
  std::vector<int> offsets_;
  std::vector<int> lengths_;
  std::vector<int> indices_;
  std::vector<T> values_;

  // labels
  std::vector<int> labelOffsets_;
  std::vector<int> labelLengths_;
  std::vector<int> labels_;

  DataLayerOpt() {}
  void loadData(const std::string &srcFile);

  inline int lengthByRecordIndex(size_t n) {
    return lengths_[n];
  }
  inline int *indicesByRecordIndex(size_t n) {
    return indices_.data() + offsets_[n];
  }
  inline T *valuesByRecordIndex(size_t n) {
    return values_.data() + offsets_[n];
  }
  inline int labelLengthByRecordIndex(size_t n) {
    return labelLengths_[n];
  }
  inline int *labelsByRecordIndex(size_t n) {
    return labels_.data() + labelOffsets_[n];

  }
};
