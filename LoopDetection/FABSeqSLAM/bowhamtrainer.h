#ifndef BOWHAMTRAINER_H
#define BOWHAMTRAINER_H

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class BOWHAMTrainer: public BOWTrainer
{
public:
    BOWHAMTrainer(double clusterSize = 0.4);
    virtual ~BOWHAMTrainer();

    // Returns trained vocabulary (i.e. cluster centers).
    virtual Mat cluster() const;
    virtual Mat cluster(const cv::Mat& descriptors) const;

protected:

    double clusterSize;
};

#endif // BOWHAMTRAINER_H
