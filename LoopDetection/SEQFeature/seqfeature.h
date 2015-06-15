#ifndef SEQFEATURE_H
#define SEQFEATURE_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class CV_EXPORTS_W SEQFeature : public Feature2D
{
public:
    CV_WRAP explicit SEQFeature(int patchSize=8);

    // Feature2D interface
    void operator ()(InputArray image, InputArray mask, vector<KeyPoint> &keypoints, OutputArray descriptors, bool useProvidedKeypoints) const;

    // DescriptorExtractor interface
    int descriptorSize() const;
    int descriptorType() const;

protected:
    int patchSize;
    FeatureDetector *detector;

    void computeImpl(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors) const;
    void detectImpl(const Mat &image, vector<KeyPoint> &keypoints, const Mat &mask) const;
};

#endif // SEQFEATURE_H
