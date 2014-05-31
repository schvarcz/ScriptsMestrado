#ifndef MATCHERCV_H
#define MATCHERCV_H
#include <iostream>
#include <string>
#include <imatcher.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class MatcherCV : public IMatcher
{
public:
    MatcherCV(string methodDetector, string methodDescriptor);

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;

    void pushBack(Mat &img);
    void matchFeatures();
private:
    Mat I1p, I1c;
    vector<KeyPoint> I1ckp, I1pkp;
    Mat I1cd, I1pd;
    FlannBasedMatcher flannMatcher;
    bool firstImg;
};
#endif // MATCHERCV_H
