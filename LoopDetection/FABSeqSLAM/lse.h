#ifndef LSE_H
#define LSE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;
class LSE
{
public:
    LSE();
    Mat model(Mat points);
    Mat fit(Mat points, Mat params, float threshold);
    Mat compute(Mat points, Mat params);
};

#endif // LSE_H
