//
//  SchvarczSLAM.h
//  SchvarczSLAM
//
//  Created by Guilherme Schvarcz Franco on 12/01/2015.
//  Copyright (c) 2015 Guilherme Schvarcz Franco. All rights reserved.
//


#ifndef SCHVACZSLAM_H
#define SCHVACZSLAM_H

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

class SchvaczSLAM
{
public:
    SchvaczSLAM();
    SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor);
    SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab, Mat BOWIDFWeights);

    void init();
    Mat apply(vector<Mat> QueryImages, vector<Mat> TestImages);
    Mat apply(VideoCapture QueryImages, VideoCapture TestImages);
    Mat calcDifferenceMatrix( vector<Mat>& QueryImages, vector<Mat>& TestImages );
    Mat calcDifferenceMatrix(VideoCapture &QueryImages, VideoCapture &TestImages);
    float calcDistance(Mat queryDescs, BFMatcher matcher);
    vector<Mat> getFeaturesDescs(VideoCapture &movie);

    pair<int, double> findMatch( Mat& diff_mat, int N, int matching_dist );
    Mat findMatches( Mat& diff_mat, int matching_dist = 10 );
    Mat findMatch2( Mat& re );
    Mat findMatches2( Mat& diff_mat );
    Mat findMatch3( Mat& re );
    Mat findMatches3( Mat& diff_mat );
    void findMatch4( Mat& re, Vector< Point > &line , bool d = false);
    Mat findMatches4( Mat& diff_mat );

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

#endif // SCHVACZSLAM_H
