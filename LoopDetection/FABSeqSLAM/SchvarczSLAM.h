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

using namespace std;
using namespace cv;

#define BOW_NORM 1
#define BOW_FREQ 2
#define BOW_TFIDF_FREQ 3
#define BOW_TFIDF_NORM 4

class SchvaczSLAM
{
public:
    SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor);
    SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab, Mat BOWIDFWeights);

    void init();
    Mat apply(vector<Mat> QueryImages, vector<Mat> TestImages);
    Mat calcDifferenceMatrix( vector<Mat>& QueryImages, vector<Mat>& TestImages );

    pair<int, double> findMatch( Mat& diff_mat, int N, int matching_dist );
    Mat findMatches( Mat& diff_mat, int matching_dist = 10 );
    Mat findMatch2( Mat& re );
    Mat findMatches2( Mat& diff_mat );
    Mat findMatch3( Mat& re );
    Mat findMatches3( Mat& diff_mat );

    Mat generateVocabulary(vector<Mat> train_set);

    Mat generateBOWImageDescs(vector<Mat> dataset, int BOW_TYPE = BOW_NORM);
    Mat getCorrespondenceMatrix(){ return occurrence; }
    void setBOWType(int BOWType){ this->BOWType = BOWType; }
    int getBOWType(){ return BOWType; }


private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Mat vocab, occurrence, BOWIDFWeights;
    int RWindow;
    float minVelocity;
    float maxVelocity;
    int BOWType;
};

#endif // SCHVACZSLAM_H
