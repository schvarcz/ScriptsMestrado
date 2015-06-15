#ifndef SEQSLAM_H
#define SEQSLAM_H

#include <seqfeature.h>

class SeqSLAM
{
public:
    SeqSLAM(int patchSize);

    void operator()(VideoCapture dataset1, VideoCapture dataset2);

protected:
    SEQFeature features;
    Mat similarityMatrix;

    virtual void detect()  = 0;
    virtual void compute() = 0;
};

class SeqSLAM1 : public SeqSLAM
{
public:

protected:
    void regularGrid(Mat image, int patchSize, vector<KeyPoint> &keypoints);
}

#endif // SEQSLAM_H
