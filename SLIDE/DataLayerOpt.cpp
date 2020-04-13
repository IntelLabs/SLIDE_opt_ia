#include <iostream>
#include <cstdlib>
#include <string.h>
#include <assert.h>
#include "DataLayerOpt.h"

void DataLayerOpt::loadData(const std::string &srcFile) {
  std::ifstream ifile(srcFile);
  std::string line;

  // parse header
  std::getline(ifile, line);
  sscanf(line.c_str(), "%ld %ld %ld\n",
         &numRecords_, &numFeatures_, &numLabels_);
  offsets_.reserve(numRecords_);
  lengths_.reserve(numRecords_);
  labelOffsets_.reserve(numRecords_);
  labelLengths_.reserve(numRecords_);

#if DEBUG_DATA_LAYER
  printf("numPoints=%ld, numFeatures=%ld, numLabels=%ld\n",
         numRecords_, numFeatures_, numLabels_);
#endif

  size_t numPointIndex = 0;
  size_t totalFeatureLength = 0;
  size_t totalLabelLength = 0;

  while (std::getline(ifile, line)) {
    char *p = (char *)line.c_str();
    // parse features
    size_t numFeatures = 0;
    char *pch_feature = strtok(p, " ");
    pch_feature = strtok(NULL, " :");
    while (pch_feature != NULL) {
      if (numFeatures % 2 == 0) {
        indices_.push_back(atoi(pch_feature));
      } else
        values_.push_back(atof(pch_feature));
      pch_feature = strtok(NULL, " :");
      numFeatures++;
    }
    lengths_[numPointIndex] = numFeatures / 2;
    offsets_[numPointIndex] = totalFeatureLength;
    totalFeatureLength += numFeatures / 2;

    // parse labels
    char *pch_label = strtok(p, ",");
    size_t numLabels = 0;
    while (pch_label != NULL) {
      labels_.push_back(atoi(pch_label));
      pch_label = strtok(NULL, ",");
      numLabels++;
    }
    labelLengths_[numPointIndex] = numLabels;
    labelOffsets_[numPointIndex] = totalLabelLength;
    totalLabelLength += numLabels;
    
#if DEBUG_DATA_LAYER
    if (numPointIndex == numRecords_ - 1) {
      std::cout << line << std::endl;
      printf("%d: featureLength=%d, totalFeatureLength=%d, label-sz=%d, "\
             "totalLabelLength=%d\n",
             numPointIndex, lengths_[numPointIndex], totalFeatureLength,
             labelLengths_[numPointIndex], totalLabelLength);
      for (int i = 0; i < lengthByRecordIndex(numPointIndex); i++) {
        float *values = valuesByRecordIndex(numPointIndex);
        int *indices = indicesByRecordIndex(numPointIndex);
        printf("point:%d=%f\n", indices[i], values[i]);
      }
      for (int i = 0; i < labelLengthByRecordIndex(numPointIndex); i++) {
        int *label = labelsByRecordIndex(numPointIndex);
        printf("label:%d\n", label[i]);
      }
    }
#endif

    numPointIndex++;
  }

  assert(numPointIndex == numRecords_);

  ifile.close();
}
