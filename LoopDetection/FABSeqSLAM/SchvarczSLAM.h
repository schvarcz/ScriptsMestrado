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


class SchvaczSLAM
{
public:
    SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor);
    SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab);

    void init();
    Mat apply(vector<Mat> QueryImages, vector<Mat> TestImages);
    Mat calcDifferenceMatrix( vector<Mat>& QueryImages, vector<Mat>& TestImages );

    pair<int, double> findMatch( Mat& diff_mat, int N, int matching_dist );
    Mat findMatches( Mat& diff_mat, int matching_dist = 10 );

    Mat generateVocabulary(vector<Mat> train_set);

    Mat generateBOWImageDescs(vector<Mat> dataset);
    Mat getCorrespondenceMatrix(){ return occurrence; }

private:
    Ptr<FeatureDetector> detector;
    Ptr<DescriptorExtractor> extractor;
    Mat vocab, occurrence;
    int RWindow;
    float minVelocity;
    float maxVelocity;
};

#endif // SCHVACZSLAM_H
