#ifndef OUTDOORSLAM_H
#define OUTDOORSLAM_H

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <math.h>

#include "ransac.h"

using namespace std;
using namespace cv;

#define BOW_NORM 1
#define BOW_FREQ 2
#define BOW_TFIDF_FREQ 3
#define BOW_TFIDF_NORM 4


class OutdoorSLAM
{
public:
public:
    OutdoorSLAM();
    OutdoorSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor);
    OutdoorSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab, Mat BOWIDFWeights);

    void init();
    Mat apply(vector<Mat> QueryImages, vector<Mat> TestImages);
    Mat apply(VideoCapture QueryImages, VideoCapture TestImages);
    Mat calcDifferenceMatrix( vector<Mat>& QueryImages, vector<Mat>& TestImages );
    Mat calcDifferenceMatrix(VideoCapture &QueryImages, VideoCapture &TestImages);
    Mat calcSequencesMatrix();

    pair<int, double> findMatch( Mat& diff_mat, int N, int matching_dist );
    Mat findMatches( Mat& diff_mat, int matching_dist = 10 );

    //Utils
    Mat vectorToMat(Vector < Point > pts);
    float lineRank(Mat img, Mat pts);
    Vector < Point > matToVector(Mat pts);
    void moveLine(Vector< Point > &line,int idx, int desv, int max);
    void draw(Mat img, Vector< Point > pts);

    Mat generateVocabulary(vector<Mat> train_set);

    Mat generateBOW(Mat frame, BOWImgDescriptorExtractor bide);
    Mat generateBOWImageDescs(VideoCapture movie, int BOW_TYPE);
    Mat generateBOWImageDescs(vector<Mat> dataset, int BOW_TYPE = BOW_NORM);
    Mat generateBOWImageDescs(Mat frame, Mat &schvarczSLAMTrainData, BOWImgDescriptorExtractor bide, int BOW_TYPE);
    Mat getCorrespondenceMatrix(){ return occurrence; }
    void setBOWType(int BOWType){ this->BOWType = BOWType; }
    int getBOWType(){ return BOWType; }


    int RWindow;
    float minVelocity;
    float maxVelocity;
    float maxVar;
    int maxHalfWindowMeanShiftSize;
private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Mat vocab, occurrence, BOWIDFWeights;
    int BOWType;
};

#endif // OUTDOORSLAM_H
