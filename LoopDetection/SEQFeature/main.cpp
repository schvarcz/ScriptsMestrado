#include <QCoreApplication>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "seqfeature.h"
#include <iostream>

using namespace cv;
using namespace std;

void regularGrid(Mat image, int patchSize, vector<KeyPoint> &keypoints)
{
    for(int col=patchSize/2; col<(int)image.cols; col+=patchSize)
        for(int row=patchSize/2; row<(int)image.rows; row+=patchSize)
            keypoints.push_back(KeyPoint(col,row,patchSize));
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    Feature2D *features;
    features = new SEQFeature(8);
    //features = Feature2D::create("SURF");

    Mat imgKpts, desc;
    vector<KeyPoint> kpts;
    Mat img1 = imread("img1.png"), img2 = imread("img2.png");
    cvtColor(img1,img1,CV_RGB2GRAY);
    cvtColor(img2,img2,CV_RGB2GRAY);
    resize(img1,img1,Size(64,32));
    //resize(img2,img2,Size(64,32));

    regularGrid(img1,8,kpts);

    //features->detect(img1,kpts);
    drawKeypoints(img1,kpts,imgKpts);

    features->compute(img1,kpts,desc);
    drawKeypoints(img1,kpts,img1);

    cout << kpts.size() << endl;
    cout << desc.rows << ", " << desc.cols << endl;
    cout << desc(Rect(0,0,10,10)) << endl;

    imshow("img",img1);
    imshow("imgKpts",imgKpts);
    waitKey();

    return a.exec();
}
