#ifndef RANSAC_H
#define RANSAC_H

#include "lse.h"

class Ransac
{
public:
    Ransac(LSE model, int threshold = 500, int iterations = 100, int minInliers = 28, int sampleSize = 4);
    Mat compute(Mat pts);
    Mat randomPts(Mat pts);
    Vector< int > randomSamples(Mat pts);

    LSE model;
    int threshold, iterations, minInliers, sampleSize;
};

#endif // RANSAC_H
