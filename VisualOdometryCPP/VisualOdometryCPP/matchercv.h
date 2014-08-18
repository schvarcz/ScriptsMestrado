#ifndef MATCHERCV_H
#define MATCHERCV_H
#include <iostream>
#include <string>
#include <imatcher.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>


using namespace cv;
using namespace std;

class MatcherCV : public IMatcher
{
public:
    MatcherCV(string methodDetector, string methodDescriptor);

    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> descriptor;

    void pushBack(Mat &img, bool replace = false);
    void matchFeatures();
private:
    Mat I1p, I1c;
    vector<KeyPoint> I1ckp, I1pkp;
    Mat I1cd, I1pd;
    FlannBasedMatcher flannMatcher;

    vector<DMatch> twoWayMatchCloser(vector<vector<DMatch> > matches1, vector<vector<DMatch> > matches2);
    vector<DMatch> twoWayMatchCloser(vector<DMatch> matches1, vector<DMatch> matches2);
    vector<DMatch> twoWayMatch(vector<vector<DMatch> > matches1, vector<vector<DMatch> > matches2);
    vector<DMatch> twoWayMatch(vector<DMatch> matches1, vector<DMatch> matches2);


    bool checkAcceptance(DMatch candidateTest, DMatch correspondentTest, float maxDist);
};
#endif // MATCHERCV_H
