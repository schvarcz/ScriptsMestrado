#ifndef SEQDESCRIPTOR_H
#define SEQDESCRIPTOR_H

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class SEQDescriptor: DescriptorExtractor
{
public:
    SEQDescriptor(int patchSize =4);

    void compute(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors) const;
    void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors ) const;
protected:
    int mPatchSize;
};

#endif // SEQDESCRIPTOR_H
