#ifndef ROBOT_H
#define ROBOT_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Robot
{
public:
    Robot();
    Mat getImage();
    void setImage(Mat newImage);
    Point2f getPosition();
    void estimateMovement(Mat newImage);
    void loopDetection();
    void relaxGraph();
    vector<Point2f> mPath;
    //MNode mKnowleadge;
    Mat mView;
};

struct MNode
{
    Point2f position;
    vector<struct MNodeTransition> nodeFrom,nodeTo;
    vector<Mat> mViews;
};

struct MNodeTransition
{
    struct MNode mOrigin, mDestiny;
    float transformation[3][3];
};

#endif // ROBOT_H
