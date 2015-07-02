#define OPENCV2P4
#include <QtCore>
#include <QThread>
#include <QDebug>
#include <QDir>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <time.h>

#ifdef OPENCV2P4
#include <opencv2/nonfree/nonfree.hpp>
#endif

#include "openfabmap.hpp"
#include "OpenSeqSLAM.h"
#include "fabseqslam.h"
#include "SchvarczSLAM.h"
#include "lse.h"
#include "ransac.h"
#include "bowhamtrainer.h"

#ifndef CLOCKS_PER_SECOND
#define CLOCKS_PER_SECOND 1000000.0
#endif
using namespace std;
using namespace cv;

/*
openFABMAP procedural functions
*/
int showFeatures(string trainPath,
                 Ptr<FeatureDetector> &detector);
int generateVocabTrainData(string trainPath,
                           string vocabTrainDataPath,
                           Ptr<FeatureDetector> &detector,
                           Ptr<DescriptorExtractor> &extractor);
int trainVocabulary(string BOWTrainerType,
                    string vocabPath,
                    string vocabTrainDataPath,
                    double clusterRadius);

int generateBOWImageDescs(string dataPath,
                          string bowImageDescPath,
                          string vocabPath,
                          Ptr<FeatureDetector> &detector,
                          Ptr<DescriptorExtractor> &extractor,
                          int minWords);

int trainChowLiuTree(string chowliutreePath,
                     string fabmapTrainDataPath,
                     double lowerInformationBound);

int openFABMAP(string testPath,
               of2c::FabMap *openFABMAP,
               string vocabPath,
               string resultsPath,
               bool addNewOnly);

void showLoopsDetections(Mat matches, string newImages,  string oldImages, Mat CorrespondenceImage, string ResultsPath,float threshold = 0.99);
/*
helper functions
*/
of2c::FabMap *generateFABMAPInstance(FileStorage &settings);
Ptr<FeatureDetector> generateDetector(FileStorage &fs);
Ptr<DescriptorExtractor> generateExtractor(FileStorage &fs);

/*
Advanced tools for keypoint manipulation. These tools are not currently in the
functional code but are available for use if desired.
*/
void drawRichKeypoints(const Mat& src, vector<KeyPoint>& kpts,
                       Mat& dst);
void filterKeypoints(vector<KeyPoint>& kpts, int maxSize = 0,
                     int maxFeatures = 0);
void sortKeypoints(vector<KeyPoint>& keypoints);

/*
shows the features detected on the training video
*/
int showFeatures(string trainPath, Ptr<FeatureDetector> &detector)
{

    //open the movie
    cv::VideoCapture movie;
    movie.open(trainPath);

    if (!movie.isOpened()) {
        cerr << trainPath << ": training movie not found" << endl;
        return -1;
    }

    cout << "Press Esc to Exit" << endl;
    Mat frame, kptsImg;

    movie.read(frame);
    vector<KeyPoint> kpts;
    int frameId = 0;
    while (movie.read(frame)) {
        detector->detect(frame, kpts);

        cout << kpts.size() << " keypoints detected...         \r";
        fflush(stdout);

        drawKeypoints(frame, kpts, kptsImg);

        imshow("Features", kptsImg);
        char name[100];
        sprintf(name,"/home/schvarcz/Desktop/features/I_%06d.png",frameId);
        imwrite(name, kptsImg);
        frameId++;
        if(waitKey(100) == 27) {
            break;
        }
    }
    cout << endl;

    destroyWindow("Features");
    return 0;
}

/*
generate the data needed to train a codebook/vocabulary for bag-of-words methods
*/
int generateVocabTrainDataPatch(string trainPath,
                           string vocabTrainDataPath,
                           Ptr<FeatureDetector> &detector,
                           Ptr<DescriptorExtractor> &extractor)
{

    //Do not overwrite any files
    ifstream checker;
    checker.open(vocabTrainDataPath.c_str());
    if(checker.is_open()) {
        cerr << vocabTrainDataPath << ": Training Data already present" <<
                     endl;
        checker.close();
        return -1;
    }

    //load training movie
    VideoCapture movie;
    movie.open(trainPath);
    if (!movie.isOpened()) {
        cerr << trainPath << ": training movie not found" << endl;
        return -1;
    }

    //extract data
    cout << "Extracting Descriptors" << endl;

    Mat vocabTrainData;
    Mat frame, descs, feats;

    Size patchSize(160,80);
    Mat patch, patch_mean, patch_stddev, temp;
    double ma, mi;

    cout.setf(ios_base::fixed);
    cout.precision(0);
    while(movie.read(frame)) {
        imshow("TFrame", frame);
        cout << "Detecting wafer mode... ";
        cvtColor(frame,feats,CV_GRAY2RGB);

        vector<KeyPoint> kpts;

        for(int y = 0; y < frame.rows; y+= patchSize.height ) {
            for(int x = 0; x < frame.cols; x+= patchSize.width ) {
                /* Extract patch */
                Rect roi(x, y, patchSize.width, patchSize.height);
                patch = frame(roi);

                minMaxIdx(patch,&mi,&ma);
                patch=255*(patch-mi)/(ma-mi);

//                /* Find the mean and std dev, calc  */
//                meanStdDev( patch, patch_mean, patch_stddev );
//                double mean_val   = patch_mean.at<double>(0, 0);
//                double stddev_val = patch_stddev.at<double>(0, 0);

//                /* Well, to avoid buffer issues, let's use double for this iteration */
//                patch.convertTo( temp, CV_64FC1 );

//                /* In Matlab 127 + x / 0.0 == 0, while in here, 127 + x / 0.0 == 127, so we need to handle that case  */
//                if( stddev_val > 0.0 ) {

//                    /* Normalize the patch */
//                    for( MatIterator_<double> itr = temp.begin<double>(); itr != temp.end<double>(); itr++ )
//                        *itr = cvRound( (*itr - mean_val) / (3*stddev_val) );
//                }
//                else
//                    temp = Scalar::all(0);

//                temp = 127*temp + 127;
//                temp.convertTo( patch, CV_8UC1 );

                vector<KeyPoint> kptsPath;
                //detect & extract features
                detector->detect(patch, kptsPath);
                extractor->compute(patch, kptsPath, descs);
                vocabTrainData.push_back(descs);
                Mat featsr;
                drawKeypoints(patch, kptsPath, featsr);
                imshow("Training Features", featsr);
            }
        }

        //add all descriptors to the training data

        //show progress
        drawKeypoints(frame, kpts, feats);
        imshow("Training Data", feats);

        cout << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                            movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%. " <<
                     vocabTrainData.rows << " descriptors         \r";
        fflush(stdout);

        if(waitKey(5) == 27) {
            destroyWindow("Training Data");
            cout << endl;
            return -1;
        }

    }
    destroyWindow("Training Data");
    cout << "Done: " << vocabTrainData.rows << " Descriptors" << endl;

    //save the training data
    FileStorage fs;
    fs.open(vocabTrainDataPath, FileStorage::WRITE);
    fs << "VocabTrainData" << vocabTrainData;
    fs.release();

    return 0;
}
/*
generate the data needed to train a codebook/vocabulary for bag-of-words methods
*/
int generateVocabTrainData(string trainPath,
                           string vocabTrainDataPath,
                           Ptr<FeatureDetector> &detector,
                           Ptr<DescriptorExtractor> &extractor)
{

    //Do not overwrite any files
    ifstream checker;
    checker.open(vocabTrainDataPath.c_str());
    if(checker.is_open()) {
        cerr << vocabTrainDataPath << ": Training Data already present" <<
                     endl;
        checker.close();
        return -1;
    }

    //load training movie
    VideoCapture movie;
    movie.open(trainPath);
    if (!movie.isOpened()) {
        cerr << trainPath << ": training movie not found" << endl;
        return -1;
    }

    //extract data
    cout << "Extracting Descriptors" << endl;

    Mat vocabTrainData;
    Mat frame, descs, feats;
    vector<KeyPoint> kpts;

    BriefDescriptorExtractor brief;

    cout.setf(ios_base::fixed);
    cout.precision(0);
    while(movie.read(frame)) {
        imshow("TFrame", frame);
        cout << "Detecting..." << endl;
        //detect & extract features
        detector->detect(frame, kpts);
        cout << "Detected" << endl;
        extractor->compute(frame, kpts, descs);
        cout << "Extracted" << endl;

        //add all descriptors to the training data
        vocabTrainData.push_back(descs);

        //show progress
        drawKeypoints(frame, kpts, feats);
        imshow("Training Data", feats);

        cout << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                            movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%. " <<
                     vocabTrainData.rows << " descriptors         \r";
        fflush(stdout);

        if(waitKey(5) == 27) {
            destroyWindow("Training Data");
            cout << endl;
            return -1;
        }

    }
    destroyWindow("Training Data");
    cout << "Done: " << vocabTrainData.rows << " Descriptors" << endl;

    //save the training data
    FileStorage fs;
    fs.open(vocabTrainDataPath, FileStorage::WRITE);
    fs << "VocabTrainData" << vocabTrainData;
    fs.release();

    return 0;
}

/*
use training data to build a codebook/vocabulary
*/
int trainVocabulary(string BOWTrainerType,
                    string vocabPath,
                    string vocabTrainDataPath,
                    double clusterRadius)
{

    //ensure not overwriting a vocabulary
    ifstream checker;
    checker.open(vocabPath.c_str());
    if(checker.is_open()) {
        cerr << vocabPath << ": Vocabulary already present" <<
                     endl;
        checker.close();
        return -1;
    }

    cout << "Loading vocabulary training data" << endl;

    FileStorage fs;

    //load in vocab training data
    fs.open(vocabTrainDataPath, FileStorage::READ);
    Mat vocabTrainData;
    fs["VocabTrainData"] >> vocabTrainData;
    if (vocabTrainData.empty()) {
        cerr << vocabTrainDataPath << ": Training Data not found" <<
                     endl;
        return -1;
    }
    fs.release();

    cout << "Performing clustering" << endl;

    //uses Modified Sequential Clustering to train a vocabulary
    BOWTrainer *trainer;
    if (BOWTrainerType == "Mahalanobis")
        trainer = new of2c::BOWMSCTrainer(clusterRadius);
    else if (BOWTrainerType == "Hamming")
        trainer = new BOWHAMTrainer(clusterRadius);

    trainer->add(vocabTrainData);
    Mat vocab = trainer->cluster();

    //save the vocabulary
    cout << "Saving vocabulary" << endl;
    fs.open(vocabPath, FileStorage::WRITE);
    fs << "Vocabulary" << vocab;
    fs.release();

    return 0;
}

/*
generate a Chow-Liu tree from FabMap Training data
*/
int trainChowLiuTree(string chowliutreePath,
                     string fabmapTrainDataPath,
                     double lowerInformationBound)
{

    FileStorage fs;

    //ensure not overwriting training data
    ifstream checker;
    checker.open(chowliutreePath.c_str());
    if(checker.is_open()) {
        cerr << chowliutreePath << ": Chow-Liu Tree already present" <<
                     endl;
        checker.close();
        return -1;
    }

    //load FabMap training data
    cout << "Loading FabMap Training Data" << endl;
    fs.open(fabmapTrainDataPath, FileStorage::READ);
    Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << endl;
        return -1;
    }
    fs.release();

    //generate the tree from the data
    cout << "Making Chow-Liu Tree" << endl;
    of2c::ChowLiuTree tree;
    tree.add(fabmapTrainData);
    Mat clTree = tree.make(lowerInformationBound);

    //save the resulting tree
    cout <<"Saving Chow-Liu Tree" << endl;
    fs.open(chowliutreePath, FileStorage::WRITE);
    fs << "ChowLiuTree" << clTree;
    fs.release();

    return 0;

}

/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/
int generateBOWImageDescs(string dataPath,
                          string bowImageDescPath,
                          string vocabPath,
                          Ptr<FeatureDetector> &detector,
                          Ptr<DescriptorExtractor> &extractor,
                          int minWords)
{

    FileStorage fs;

    //ensure not overwriting training data
    ifstream checker;
    checker.open(bowImageDescPath.c_str());
    if(checker.is_open()) {
        cerr << bowImageDescPath << ": FabMap Training/Testing Data "
                     "already present" << endl;
        checker.close();
        return -1;
    }

    //load vocabulary
    cout << "Loading Vocabulary" << endl;
    fs.open(vocabPath, FileStorage::READ);
    Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        cerr << vocabPath << ": Vocabulary not found" << endl;
        return -1;
    }
    fs.release();

    //use a FLANN matcher to generate bag-of-words representations
    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //load movie
    VideoCapture movie;
    movie.open(dataPath);

    if(!movie.isOpened()) {
        cerr << dataPath << ": movie not found" << endl;
        return -1;
    }

    //extract image descriptors
    Mat fabmapTrainData;
    cout << "Extracting Bag-of-words Image Descriptors" << endl;
    cout.setf(ios_base::fixed);
    cout.precision(0);

    ofstream maskw;

    if(minWords) {
        maskw.open(string(bowImageDescPath + "mask.txt").c_str());
    }

    Mat frame, bow;
    vector<KeyPoint> kpts;

    while(movie.read(frame)) {
        detector->detect(frame, kpts);
        bide.compute(frame, kpts, bow);

        if(minWords) {
            //writing a mask file
            if(countNonZero(bow) < minWords) {
                //frame masked
                maskw << "0" << endl;
            } else {
                //frame accepted
                maskw << "1" << endl;
                fabmapTrainData.push_back(bow);
            }
        } else {
            fabmapTrainData.push_back(bow);
        }

        cout << 100.0 * (movie.get(CV_CAP_PROP_POS_FRAMES) /
                              movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%    \r";
        fflush(stdout);
    }
    cout << "Done                                       " << endl;

    movie.release();

    //save training data
    fs.open(bowImageDescPath, FileStorage::WRITE);
    fs << "BOWImageDescs" << fabmapTrainData;
    fs.release();

    return 0;
}

/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/
int generateBOWIDFWeights(string bowImageDescPath,
                             string bowIDFWeightsPath)
{

    FileStorage fs;

    //ensure not overwriting training data
    ifstream checker;
    checker.open(bowIDFWeightsPath.c_str());
    if(checker.is_open()) {
        cerr << bowIDFWeightsPath << ": FabMap BOW IDF Weights Data already present" << endl;
        checker.close();
        return -1;
    }

    //load vocabulary
    cout << "Loading BOWImageDesc" << endl;
    fs.open(bowImageDescPath, FileStorage::READ);
    Mat bowImageDesc;
    fs["BOWImageDescs"] >> bowImageDesc;
    if (bowImageDesc.empty()) {
        cerr << bowImageDescPath << ": BOWImageDesc not found" << endl;
        return -1;
    }
    fs.release();


    //extract image descriptors
    Mat BOWIDFWeights;
    cout << "Extracting Bag-of-words IDF Weights" << endl;

    cout << "Cols " << bowImageDesc.cols << " - Rows " << bowImageDesc.rows << endl;

    for(int indexWord = 0; indexWord < bowImageDesc.cols; indexWord++)
    {
        int occurrence = 0;
        for(int indexDocument = 0; indexDocument < bowImageDesc.rows; indexDocument++)
        {
            if (bowImageDesc.at<float>(indexDocument,indexWord) != 0)
                occurrence++;
        }
        cout << "\r"<<  indexWord << "° Word " << 100.0*indexWord/bowImageDesc.cols << "%                  " << occurrence;
        fflush(stdout);
        float idf = 0;
        if (occurrence != 0)
            idf = log(bowImageDesc.rows/occurrence);


        BOWIDFWeights.push_back(idf);
    }

    cout << "Done                                       " << endl;


    //save training data
    fs.open(bowIDFWeightsPath, FileStorage::WRITE);
    fs << "BOWIDFWeights" << BOWIDFWeights;
    fs.release();

    return 0;
}

/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/
Mat generateBOWImageDescs(string dataPath,
                              Mat vocab,
                              Ptr<FeatureDetector> &detector,
                              Ptr<DescriptorExtractor> &extractor)
{

    //use a FLANN matcher to generate bag-of-words representations
    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //load movie
    VideoCapture movie;
    movie.open(dataPath);

    if(!movie.isOpened()) {
        cerr << dataPath << ": movie not found" << endl;
        return Mat();
    }

    //extract image descriptors
    Mat fabmapTrainData;
    cout << "Extracting Bag-of-words Image Descriptors" << endl;

    Mat frame, bow;
    vector<KeyPoint> kpts;

    while(movie.read(frame)) {
        detector->detect(frame, kpts);
        bide.compute(frame, kpts, bow);

        fabmapTrainData.push_back(bow);

        cout << "\r " << kpts.size() << " keypoints detected... " << 100.0 * (movie.get(CV_CAP_PROP_POS_FRAMES) /
                              movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%             ";
        fflush(stdout);

    }
    cout << "Done                                       " << endl;

    movie.release();

    return fabmapTrainData;
}
/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/

Mat generateBOW(Mat frame, BOWImgDescriptorExtractor bide,
                             Ptr<FeatureDetector> &detector)
{
    Mat result = frame.clone();
    if ( result.channels() > 1 )
        cvtColor( result, result, CV_BGR2GRAY );

    Size patchSize(160,80);
    Mat patch, patch_mean, patch_stddev, temp;
    double mi, ma;

    Mat bowReturn = Mat::zeros(1, bide.getVocabulary().rows, CV_32F);


    for(int y = 0; y < result.rows; y+= patchSize.height ) {
        for(int x = 0; x < result.cols; x+= patchSize.width ) {
            /* Extract patch */
            patch = result(Rect(x, y, patchSize.width, patchSize.height));


            minMaxIdx(patch,&mi,&ma);
            patch=255*(patch-mi)/(ma-mi);

//            /* Find the mean and std dev, calc  */
//            meanStdDev( patch, patch_mean, patch_stddev );
//            double mean_val   = patch_mean.at<double>(0, 0);
//            double stddev_val = patch_stddev.at<double>(0, 0);

//            /* Well, to avoid buffer issues, let's use double for this iteration */
//            patch.convertTo( temp, CV_64FC1 );

//            /* In Matlab 127 + x / 0.0 == 0, while in here, 127 + x / 0.0 == 127, so we need to handle that case  */
//            if( stddev_val > 0.0 ) {

//                /* Normalize the patch */
//                for( MatIterator_<double> itr = temp.begin<double>(); itr != temp.end<double>(); itr++ )
//                    *itr = cvRound( (*itr - mean_val) / (3*stddev_val) );
//            }
//            else
//                temp = Scalar::all(0);

//            temp = 127*temp + 127;
//            temp.convertTo( patch, CV_8UC1 );

            vector<KeyPoint> kpts;
            Mat bow;

            detector->detect(patch, kpts);
            bide.compute(patch, kpts, bow);

            if (bow.cols != 0)
                bowReturn+= bow;
        }
    }

    return bowReturn;
}

/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/
int generateBOWImageDescsWafer(string dataPath,
                          string bowImageDescPath,
                          string vocabPath,
                          Ptr<FeatureDetector> &detector,
                          Ptr<DescriptorExtractor> &extractor,
                          int minWords)
{

    FileStorage fs;

    //ensure not overwriting training data
    ifstream checker;
    checker.open(bowImageDescPath.c_str());
    if(checker.is_open()) {
        cerr << bowImageDescPath << ": FabMap Training/Testing Data "
                     "already present" << endl;
        checker.close();
        return -1;
    }

    //load vocabulary
    cout << "Loading Vocabulary" << endl;
    fs.open(vocabPath, FileStorage::READ);
    Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        cerr << vocabPath << ": Vocabulary not found" << endl;
        return -1;
    }
    fs.release();

    //use a FLANN matcher to generate bag-of-words representations
    Ptr<DescriptorMatcher> matcher =
            DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //load movie
    VideoCapture movie;
    movie.open(dataPath);

    if(!movie.isOpened()) {
        cerr << dataPath << ": movie not found" << endl;
        return -1;
    }

    //extract image descriptors
    Mat fabmapTrainData;
    cout << "Extracting Bag-of-words Image Descriptors" << endl;
    cout.setf(ios_base::fixed);
    cout.precision(0);

    ofstream maskw;

    if(minWords) {
        maskw.open(string(bowImageDescPath + "mask.txt").c_str());
    }

    Mat frame, bow;

    while(movie.read(frame)) {
        bow = generateBOW(frame, bide, detector);

        if(minWords) {
            //writing a mask file
            if(countNonZero(bow) < minWords) {
                //frame masked
                maskw << "0" << endl;
            } else {
                //frame accepted
                maskw << "1" << endl;
                fabmapTrainData.push_back(bow);
            }
        } else {
            fabmapTrainData.push_back(bow);
        }

        cout << 100.0 * (movie.get(CV_CAP_PROP_POS_FRAMES) /
                              movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%    \r";
        fflush(stdout);
    }
    cout << "Done                                       " << endl;

    movie.release();

    //save training data
    fs.open(bowImageDescPath, FileStorage::WRITE);
    fs << "BOWImageDescs" << fabmapTrainData;
    fs.release();

    return 0;
}

/*
Run FabMap on a test dataset
*/

int openFABMAP(string testPath,
               of2c::FabMap *fabmap,
               string vocabPath,
               string resultsPath,
               bool addNewOnly)
{
    FileStorage fs;

    //ensure not overwriting results
    ifstream checker;
    checker.open(resultsPath.c_str());
    if(checker.is_open()) {
        cerr << resultsPath << ": Results already present" << endl;
        checker.close();
        return -1;
    }

    //load the vocabulary
    cout << "Loading Vocabulary" << endl;
    fs.open(vocabPath, FileStorage::READ);
    Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        cerr << vocabPath << ": Vocabulary not found" << endl;
        return -1;
    }
    fs.release();

    //load the test data
    fs.open(testPath, FileStorage::READ);
    Mat testImageDescs;
    fs["BOWImageDescs"] >> testImageDescs;
    if(testImageDescs.empty()) {
        cerr << testPath << ": Test data not found" << endl;
        return -1;
    }
    fs.release();

    //running openFABMAP
    cout << "Running openFABMAP" << endl;
    vector<of2c::IMatch> matches;
    vector<of2c::IMatch>::iterator l;



    Mat confusion_mat(testImageDescs.rows, testImageDescs.rows, CV_64FC1);
    confusion_mat = Scalar(0); // init to 0's


    if (!addNewOnly) {

        //automatically comparing a whole dataset
        fabmap->compare(testImageDescs, matches, true);

        for(l = matches.begin(); l != matches.end(); l++) {
            if(l->imgIdx < 0) {
                confusion_mat.at<double>(l->queryIdx, l->queryIdx) = l->match;

            } else {
                confusion_mat.at<double>(l->queryIdx, l->imgIdx) = l->match;
            }
        }

    } else {

        //criteria for adding locations used
        for(int i = 0; i < testImageDescs.rows; i++) {
            matches.clear();
            //compare images individually
            fabmap->compare(testImageDescs.row(i), matches);

            bool new_place_max = true;
            for(l = matches.begin(); l != matches.end(); l++) {

                if(l->imgIdx < 0) {
                    //add the new place to the confusion matrix 'diagonal'
                    confusion_mat.at<double>(i, matches.size()-1) = l->match;

                } else {
                    //add the score to the confusion matrix
                    confusion_mat.at<double>(i, l->imgIdx) = l->match;
                }

                //test for new location maximum
                if(l->match > matches.front().match) {
                    new_place_max = false;
                }
            }

            if(new_place_max) {
                fabmap->add(testImageDescs.row(i));
            }
        }
    }

    //save the result as plain text for ease of import to Matlab
    ofstream writer(resultsPath.c_str());
    for(int i = 0; i < confusion_mat.rows; i++) {
        for(int j = 0; j < confusion_mat.cols; j++) {
            writer << confusion_mat.at<double>(i, j) << " ";
        }
        writer << endl;
    }
    writer.close();

    imwrite("results.bmp",confusion_mat);

    return 0;

}

/*
Run FabMap on a test dataset
*/

vector<of2c::IMatch> openFABMAP(Mat queryImageDescs,
               Mat testImageDescs,
               of2c::FabMap *fabmap)
{

    //running openFABMAP
    cout << "Running openFABMAP" << endl;
    vector<of2c::IMatch> matches;

    //automatically comparing a whole dataset
    fabmap->compare(queryImageDescs, testImageDescs, matches);

    return matches;
}


int openFABMAP(string TestPath,
               string QueryPath,
               of2c::FabMap *fabmap,
               string vocabPath,
               string CorrespondenceImageResults,
               string ResultsPath,
               Ptr<FeatureDetector> detector,
               Ptr<DescriptorExtractor> extractor)
{

    //load vocabulary
    FileStorage fs;
    cout << "Loading Vocabulary" << endl;
    fs.open(vocabPath, FileStorage::READ);
    Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        cerr << vocabPath << ": Vocabulary not found" << endl;
        return -1;
    }
    fs.release();

    const clock_t begin_time = clock();
    cout << "Generating BOW for test images: " << TestPath << endl;
    Mat BOWTest = generateBOWImageDescs(TestPath, vocab, detector, extractor);
    cout << "Generating BOW for query images: " << QueryPath << endl;
    Mat BOWQuery = generateBOWImageDescs(QueryPath, vocab, detector, extractor);

    if(fabmap) {
        vector<of2c::IMatch> matches = openFABMAP(BOWQuery,
                                BOWTest,
                                fabmap);

        cout << "FABMap Total time: " << ((clock()- begin_time)/CLOCKS_PER_SECOND) << endl;

        Mat confusion_mat(BOWQuery.rows, BOWTest.rows, CV_32F,0.0);

        Mat cvMatches(BOWQuery.rows,2 ,CV_32F, numeric_limits<float>::max());
        for(vector<of2c::IMatch>::iterator l = matches.begin(); l != matches.end(); l++) {

            if ((l->match>0.01) && (l->imgIdx != -1))
            {
                confusion_mat.at<float>(l->queryIdx, l->imgIdx) = l->match;
                cvMatches.at<float>(l->queryIdx,0) = float(l->imgIdx);
                cvMatches.at<float>(l->queryIdx,1) = float(l->match);
                cout << "Bleu: " << l->queryIdx << " - " << l->imgIdx << " - " << l->match << " " << cvMatches.at<float>(l->queryIdx,1) << endl;
            }
        }
        cout << cvMatches;
        cout << "Saving results: " << CorrespondenceImageResults << endl;
        double ma, mi;
        minMaxLoc(confusion_mat,&mi,&ma);
        confusion_mat = 255*(confusion_mat - mi)/(ma-mi);
        imwrite(CorrespondenceImageResults,confusion_mat);


        showLoopsDetections(cvMatches.t(), QueryPath,  TestPath, confusion_mat, ResultsPath, 2.0);

    }

    return 0;
}

/*
generates a feature detector based on options in the settings file
*/
Ptr<FeatureDetector> generateDetector(FileStorage &fs) {

    //create common feature detector and descriptor extractor
    string detectorMode = fs["FeatureOptions"]["DetectorMode"];
    string detectorType = fs["FeatureOptions"]["DetectorType"];
    Ptr<FeatureDetector> detector = NULL;
    if(detectorMode == "ADAPTIVE") {

        if(detectorType != "STAR" && detectorType != "SURF" &&
                detectorType != "FAST") {
            cerr << "Adaptive Detectors only work with STAR, SURF "
                         "and FAST" << endl;
        } else {

            detector = new DynamicAdaptedFeatureDetector(
                        AdjusterAdapter::create(detectorType),
                        fs["FeatureOptions"]["Adaptive"]["MinFeatures"],
                        fs["FeatureOptions"]["Adaptive"]["MaxFeatures"],
                        fs["FeatureOptions"]["Adaptive"]["MaxIters"]);
        }

    } else if(detectorMode == "STATIC") {
        if(detectorType == "STAR") {

            detector = new StarFeatureDetector(
                        fs["FeatureOptions"]["StarDetector"]["MaxSize"],
                        fs["FeatureOptions"]["StarDetector"]["Response"],
                        fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
                        fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
                        fs["FeatureOptions"]["StarDetector"]["Suppression"]);

        } else if(detectorType == "FAST") {

            detector = new FastFeatureDetector(
                        fs["FeatureOptions"]["FastDetector"]["Threshold"],
                        (int)fs["FeatureOptions"]["FastDetector"]
                        ["NonMaxSuppression"] > 0);

        } else if(detectorType == "SURF") {

#ifdef OPENCV2P4
            detector = new SURF(
                        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                        fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                        fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                        (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                        (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

#else
            detector = new SurfFeatureDetector(
                        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                        fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                        fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                        (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
#endif
        } else if(detectorType == "SIFT") {
#ifdef OPENCV2P4
            detector = new SIFT(
                        fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                        fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                        fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                        fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                        fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#else
            detector = new SiftFeatureDetector(
                        fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                        fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"]);
#endif
        } else if(detectorType == "MSER") {

            detector = new MserFeatureDetector(
                        fs["FeatureOptions"]["MSERDetector"]["Delta"],
                        fs["FeatureOptions"]["MSERDetector"]["MinArea"],
                        fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
                        fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
                        fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
                        fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
                        fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
                        fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
                        fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);

        } else if (detectorType == "ORB") {
            int scoreType = ORB::HARRIS_SCORE;
            string ScoreType = fs["FeatureOptions"]["OrbDetector"]["ScoreType"];
            if (ScoreType == "HARRIS")
            {
                scoreType = ORB::HARRIS_SCORE;
            }
            else if (ScoreType == "FAST")
            {
                scoreType = ORB::FAST_SCORE;
            }

            detector = new ORB((int)fs["FeatureOptions"]["OrbDetector"]["NumFeatures"],
                                (float)fs["FeatureOptions"]["OrbDetector"]["ScaleFactor"],
                                (int)fs["FeatureOptions"]["OrbDetector"]["NumLevels"],
                                (int)fs["FeatureOptions"]["OrbDetector"]["EdgeThreshold"],
                                (int)fs["FeatureOptions"]["OrbDetector"]["FirstLevel"],
                                (int)fs["FeatureOptions"]["OrbDetector"]["WTAK"],
                                scoreType,
                                (int)fs["FeatureOptions"]["OrbDetector"]["PatchSize"]);
        } else if (detectorType == "BRISK") {
            detector = new BRISK(
                        (int)fs["FeatureOptions"]["BriskDetector"]["Threshold"],
                        (int)fs["FeatureOptions"]["BriskDetector"]["Octaves"],
                        (float)fs["FeatureOptions"]["BriskDetector"]["PatternScale"]);
        } else {
            cerr << "Could not create detector class. Specify detector "
                         "options in the settings file" << endl;
        }
    } else {
        cerr << "Could not create detector class. Specify detector "
                     "mode (static/adaptive) in the settings file" << endl;
    }

    return detector;

}

/*
generates a feature detector based on options in the settings file
*/
Ptr<DescriptorExtractor> generateExtractor(FileStorage &fs)
{
    string extractorType = fs["FeatureOptions"]["ExtractorType"];
    Ptr<DescriptorExtractor> extractor = NULL;
    if(extractorType == "SIFT") {
#ifdef OPENCV2P4
        extractor = new SIFT(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                    fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                    fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#else
        extractor = new SiftDescriptorExtractor();
#endif

    } else if(extractorType == "SURF") {

#ifdef OPENCV2P4
        extractor = new SURF(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                    (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                    (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

#else
        extractor = new SurfDescriptorExtractor(
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                    (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                    (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
#endif

    } else if (extractorType == "BRIEF") {
        extractor = new BriefDescriptorExtractor((int)fs["FeatureOptions"]["BriefExtractor"]["Bytes"]);
    } else if (extractorType == "BRISK") {
        extractor = new BRISK(
                    (int)fs["FeatureOptions"]["BriskDetector"]["Threshold"],
                    (int)fs["FeatureOptions"]["BriskDetector"]["Octaves"],
                    (float)fs["FeatureOptions"]["BriskDetector"]["PatternScale"]);
    } else if (extractorType == "ORB") {
        int scoreType = ORB::HARRIS_SCORE;
        string ScoreType = fs["FeatureOptions"]["OrbDetector"]["ScoreType"];
        if (ScoreType == "HARRIS")
        {
            scoreType = ORB::HARRIS_SCORE;
        }
        else if (ScoreType == "FAST")
        {
            scoreType = ORB::FAST_SCORE;
        }

        extractor = new ORB((int)fs["FeatureOptions"]["OrbDetector"]["NumFeatures"],
                            (float)fs["FeatureOptions"]["OrbDetector"]["ScaleFactor"],
                            (int)fs["FeatureOptions"]["OrbDetector"]["NumLevels"],
                            (int)fs["FeatureOptions"]["OrbDetector"]["EdgeThreshold"],
                            (int)fs["FeatureOptions"]["OrbDetector"]["FirstLevel"],
                            (int)fs["FeatureOptions"]["OrbDetector"]["WTAK"],
                            scoreType,
                            (int)fs["FeatureOptions"]["OrbDetector"]["PatchSize"]);
    } else if (extractorType == "FREAK") {
        extractor = new FREAK(
                    (string)fs["FeatureOptions"]["FreakExtractor"]["OrientationNormalized"] == "True",
                    (string)fs["FeatureOptions"]["FreakExtractor"]["ScaleNormalized"] == "True",
                    (float)fs["FeatureOptions"]["FreakExtractor"]["PatternScale"],
                    (int)fs["FeatureOptions"]["FreakExtractor"]["NumOctaves"]);
    } else {
        cerr << "Could not create Descriptor Extractor. Please specify "
                     "extractor type in settings file" << endl;
    }

    return extractor;

}


int generatorBOWType(string BOWSType)
{
    if (BOWSType == "BOW_NORM")
        return BOW_NORM;
    if (BOWSType == "BOW_FREQ")
        return BOW_FREQ;
    if (BOWSType == "BOW_TFIDF_FREQ")
        return BOW_TFIDF_FREQ;
    if (BOWSType == "BOW_TFIDF_NORM")
        return BOW_TFIDF_NORM;
}


/*
create an instance of a FabMap class with the options given in the settings file
*/
of2c::FabMap *generateFABMAPInstance(FileStorage &settings)
{

    FileStorage fs;

    //load FabMap training data
    string fabmapTrainDataPath = settings["FilePaths"]["TrainImagDesc"];
    string chowliutreePath = settings["FilePaths"]["ChowLiuTree"];

    cout << "Loading FabMap Training Data" << endl;
    fs.open(fabmapTrainDataPath, FileStorage::READ);
    Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << endl;
        return NULL;
    }
    fs.release();

    //load a chow-liu tree
    cout << "Loading Chow-Liu Tree" << endl;
    fs.open(chowliutreePath, FileStorage::READ);
    Mat clTree;
    fs["ChowLiuTree"] >> clTree;
    if (clTree.empty()) {
        cerr << chowliutreePath << ": Chow-Liu tree not found" <<
                     endl;
        return NULL;
    }
    fs.release();

    //create options flags
    string newPlaceMethod =
            settings["openFabMapOptions"]["NewPlaceMethod"];
    string bayesMethod = settings["openFabMapOptions"]["BayesMethod"];
    int simpleMotionModel = settings["openFabMapOptions"]["SimpleMotion"];
    int options = 0;
    if(newPlaceMethod == "Sampled") {
        options |= of2c::FabMap::SAMPLED;
    } else {
        options |= of2c::FabMap::MEAN_FIELD;
    }
    if(bayesMethod == "ChowLiu") {
        options |= of2c::FabMap::CHOW_LIU;
    } else {
        options |= of2c::FabMap::NAIVE_BAYES;
    }
    if(simpleMotionModel) {
        options |= of2c::FabMap::MOTION_MODEL;
    }

    of2c::FabMap *fabmap;

    //create an instance of the desired type of FabMap
    string fabMapVersion = settings["openFabMapOptions"]["FabMapVersion"];
    if(fabMapVersion == "FABMAP1") {
        fabmap = new of2c::FabMap1(clTree,
                                   settings["openFabMapOptions"]["PzGe"],
                                   settings["openFabMapOptions"]["PzGne"],
                                   options,
                                   settings["openFabMapOptions"]["NumSamples"]);
    } else if(fabMapVersion == "FABMAPLUT") {
        fabmap = new of2c::FabMapLUT(clTree,
                                     settings["openFabMapOptions"]["PzGe"],
                                     settings["openFabMapOptions"]["PzGne"],
                                     options,
                                     settings["openFabMapOptions"]["NumSamples"],
                                     settings["openFabMapOptions"]["FabMapLUT"]["Precision"]);
    } else if(fabMapVersion == "FABMAPFBO") {
        fabmap = new of2c::FabMapFBO(clTree,
                                     settings["openFabMapOptions"]["PzGe"],
                                     settings["openFabMapOptions"]["PzGne"],
                                     options,
                                     settings["openFabMapOptions"]["NumSamples"],
                                     settings["openFabMapOptions"]["FabMapFBO"]["RejectionThreshold"],
                                     settings["openFabMapOptions"]["FabMapFBO"]["PsGd"],
                                     settings["openFabMapOptions"]["FabMapFBO"]["BisectionStart"],
                                     settings["openFabMapOptions"]["FabMapFBO"]["BisectionIts"]);
    } else if(fabMapVersion == "FABMAP2") {
        fabmap = new of2c::FabMap2(clTree,
                                   settings["openFabMapOptions"]["PzGe"],
                                   settings["openFabMapOptions"]["PzGne"],
                                   options);
    } else {
        cerr << "Could not identify openFABMAPVersion from settings"
                     " file" << endl;
        return NULL;
    }

    //add the training data for use with the sampling method
    fabmap->addTraining(fabmapTrainData);

    return fabmap;

}



/*
draws keypoints to scale with coloring proportional to feature strength
*/
void drawRichKeypoints(const Mat& src, vector<KeyPoint>& kpts, Mat& dst) {

    Mat grayFrame;
    cvtColor(src, grayFrame, CV_RGB2GRAY);
    cvtColor(grayFrame, dst, CV_GRAY2RGB);

    if (kpts.size() == 0) {
        return;
    }

    vector<KeyPoint> kpts_cpy, kpts_sorted;

    kpts_cpy.insert(kpts_cpy.end(), kpts.begin(), kpts.end());

    double maxResponse = kpts_cpy.at(0).response;
    double minResponse = kpts_cpy.at(0).response;

    while (kpts_cpy.size() > 0) {

        double maxR = 0.0;
        unsigned int idx = 0;

        for (unsigned int iii = 0; iii < kpts_cpy.size(); iii++) {

            if (kpts_cpy.at(iii).response > maxR) {
                maxR = kpts_cpy.at(iii).response;
                idx = iii;
            }

            if (kpts_cpy.at(iii).response > maxResponse) {
                maxResponse = kpts_cpy.at(iii).response;
            }

            if (kpts_cpy.at(iii).response < minResponse) {
                minResponse = kpts_cpy.at(iii).response;
            }
        }

        kpts_sorted.push_back(kpts_cpy.at(idx));
        kpts_cpy.erase(kpts_cpy.begin() + idx);

    }

    int thickness = 1;
    Point center;
    Scalar colour;
    int red = 0, blue = 0, green = 0;
    int radius;
    double normalizedScore;

    if (minResponse == maxResponse) {
        colour = CV_RGB(255, 0, 0);
    }

    for (int iii = kpts_sorted.size()-1; iii >= 0; iii--) {

        if (minResponse != maxResponse) {
            normalizedScore = pow((kpts_sorted.at(iii).response - minResponse) / (maxResponse - minResponse), 0.25);
            red = int(255.0 * normalizedScore);
            green = int(255.0 - 255.0 * normalizedScore);
            colour = CV_RGB(red, green, blue);
        }

        center = kpts_sorted.at(iii).pt;
        center.x *= 16;
        center.y *= 16;

        radius = (int)(16.0 * ((double)(kpts_sorted.at(iii).size)/2.0));

        if (radius > 0) {
            circle(dst, center, radius, colour, thickness, CV_AA, 4);
        }

    }

}

/*
Removes surplus features and those with invalid size
*/
void filterKeypoints(vector<KeyPoint>& kpts, int maxSize, int maxFeatures) {

    if (maxSize == 0) {
        return;
    }

    sortKeypoints(kpts);

    for (unsigned int iii = 0; iii < kpts.size(); iii++) {

        if (kpts.at(iii).size > float(maxSize)) {
            kpts.erase(kpts.begin() + iii);
            iii--;
        }
    }

    if ((maxFeatures != 0) && ((int)kpts.size() > maxFeatures)) {
        kpts.erase(kpts.begin()+maxFeatures, kpts.end());
    }

}

/*
Sorts keypoints in descending order of response (strength)
*/
void sortKeypoints(vector<KeyPoint>& keypoints) {

    if (keypoints.size() <= 1) {
        return;
    }

    vector<KeyPoint> sortedKeypoints;

    // Add the first one
    sortedKeypoints.push_back(keypoints.at(0));

    for (unsigned int i = 1; i < keypoints.size(); i++) {

        unsigned int j = 0;
        bool hasBeenAdded = false;

        while ((j < sortedKeypoints.size()) && (!hasBeenAdded)) {

            if (abs(keypoints.at(i).response) > abs(sortedKeypoints.at(j).response)) {
                sortedKeypoints.insert(sortedKeypoints.begin() + j, keypoints.at(i));

                hasBeenAdded = true;
            }

            j++;
        }

        if (!hasBeenAdded) {
            sortedKeypoints.push_back(keypoints.at(i));
        }

    }

    keypoints.swap(sortedKeypoints);

}

/**
 * Load the dataset
 **/
vector<Mat> loadDataset( string path ) {
    QString qPath(path.c_str());
    vector<Mat> images;

    QDir diretory(qPath);
    QStringList filtro;
    filtro << "*.png";

    QStringList imageFiles = diretory.entryList(filtro);

    for(QStringList::Iterator it = imageFiles.begin(); it != imageFiles.end();it++)
    {
        QString temp = qPath + *it;
        Mat image = imread( temp.toStdString().c_str() );
        images.push_back( image );
    }

    return images;
}

vector<Mat> loadDatasetFromVideo( string path ) {
    vector<Mat> images;

    VideoCapture movie;
    movie.open(path);

    if (!movie.isOpened()) {
        cerr << path << ": training movie not found" << endl;
        return images;
    }

    //cout << "Loading video: " << path << endl;
    Mat frame;

    while (movie.read(frame)) {

        //cout << "\r " << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
        //                        movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%         ";
        fflush(stdout);

        Mat save;
        frame.copyTo(save);
        //cvtColor(frame,save,CV_RGB2GRAY);
        images.push_back( save );
    }
    //cout << "Done" << endl;

    return images;
}

VideoCapture loadDatasetVideo( string path ) {
    VideoCapture movie;
    movie.open(path);

    if (!movie.isOpened()) {
        cerr << path << ": training movie not found" << endl;
        return movie;
    }

    //cout << "Loading video: " << path << endl;
    return movie;
}

void showLoopsDetections(Mat matches, vector<Mat> newImages, vector<Mat> oldImages, Mat CorrespondenceImage, string ResultsPath,float threshold = 0.99)
{
    double ma,mi;

    minMaxLoc(CorrespondenceImage,&mi,&ma);
    CorrespondenceImage = 255*(CorrespondenceImage-mi)/(ma-mi);
    Mat corres;
    CorrespondenceImage.convertTo(corres,CV_8U);
    cvtColor(corres,CorrespondenceImage,CV_GRAY2RGB);

    CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(255, 0, 0) );
    namedWindow("");
    moveWindow("", 0, 0);

    char temp[100], name[255];

    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);

    Mat appended( newImages[0].rows,newImages[0].cols*2, newImages[0].type(), Scalar(0) );

    for( uint x = 0; x < newImages.size(); x++ ) {

        cout << "\r Image " << x << "/" << newImages.size() << " (" << 100.0*float(x)/newImages.size()<< "%)      ";

        int index = static_cast<int>(index_ptr[x]);

        //cout << index << endl;
        /* Append the images together */
        appended.setTo(Scalar(0));

        newImages[x].copyTo( Mat(appended, Rect(0, 0, appended.cols/2, appended.rows) ));

        if( score_ptr[x] < threshold )
        {
            oldImages[index].copyTo( Mat(appended, Rect(appended.cols/2, 0, appended.cols/2, appended.rows) ));
            circle(CorrespondenceImage,Point(index,x),1,Scalar(255,0,0),-1);
        }
        else
            circle(CorrespondenceImage,Point(index,x),1,Scalar(0,0,255),-1);

        /* The lower the score, the lower the differences between images */
        if( score_ptr[x] < threshold )
            sprintf( temp, "Old image [%03d]", index );
        else
            sprintf( temp, "Old image [None]" );

        addText( appended, temp, Point( appended.cols/2 + 20, 20 ), font );

        sprintf( temp, "New image [%03d]", x );
        addText( appended, temp, Point( 10, 20 ), font );


        cout << static_cast<float>(score_ptr[x]) << " - " << threshold << endl;
        if( score_ptr[x] < threshold )
        {
            sprintf( name, "%s/I_new_%06d_old_%06d_%.3f.png", ResultsPath.c_str(), x, index, score_ptr[x] );
            imwrite(name,appended);
        }

        //imshow( "", appended );
        //imshow("matches", CorrespondenceImage);
        //waitKey(500);
        fflush(stdout);
    }
    sprintf( name, "%s/matches.png", ResultsPath.c_str());
    imwrite(name,CorrespondenceImage);
}

Mat getFrameFromVideo(VideoCapture movie, int frame)
{
    if(movie.set(CV_CAP_PROP_POS_FRAMES,frame))
    {
        Mat frame;
        movie.retrieve(frame);
        imshow("frame",frame);
        waitKey();
        return frame;
    }
}

Mat getFrameFromFile(string path, int frame)
{
    char name[255];
    sprintf(name,path.c_str(),frame);
    return imread(name);
}

void showLoopsDetections(Mat matches, string newImages,  string oldImages, Mat CorrespondenceImage, string ResultsPath,float threshold)
{
    double ma,mi;

    minMaxLoc(CorrespondenceImage,&mi,&ma);
    CorrespondenceImage = 255*(CorrespondenceImage-mi)/(ma-mi);
    Mat corres;
    CorrespondenceImage.convertTo(corres,CV_8U);
    cvtColor(corres,CorrespondenceImage,CV_GRAY2RGB);

    CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(255, 0, 0) );
    namedWindow("");
    moveWindow("", 0, 0);

    char temp[100], name[255];

    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);

    Mat appended( getFrameFromFile(newImages, 0).rows,getFrameFromFile(newImages, 0).cols*2, getFrameFromFile(newImages, 0).type(), Scalar(0) );

    uint sizeDataset = loadDatasetVideo(newImages).get(CV_CAP_PROP_FRAME_COUNT);
    //cout << sizeDataset << endl;

    for( uint x = 0; x < sizeDataset; x++ ) {

        cout << "\r Image " << x << "/" << sizeDataset << " (" << 100.0*float(x)/sizeDataset<< "%),  ";

        int index = static_cast<int>(index_ptr[x]);

        //cout << index << endl;
        /* Append the images together */
        appended.setTo(Scalar(0));

        getFrameFromFile(newImages, x).copyTo( Mat(appended, Rect(0, 0, appended.cols/2, appended.rows) ));

        if( score_ptr[x] < threshold )
        {
            getFrameFromFile(oldImages, index).copyTo( Mat(appended, Rect(appended.cols/2, 0, appended.cols/2, appended.rows) ));
            circle(CorrespondenceImage,Point(index,x),1,Scalar(255,0,0),-1);
        }
        else
            circle(CorrespondenceImage,Point(index,x),1,Scalar(0,0,255),-1);

        /* The lower the score, the lower the differences between images */
        if( score_ptr[x] < threshold )
            sprintf( temp, "Old image [%03d]", index );
        else
            sprintf( temp, "Old image [None]" );

        addText( appended, temp, Point( appended.cols/2 + 20, 20 ), font );

        sprintf( temp, "New image [%03d]", x );
        addText( appended, temp, Point( 10, 20 ), font );


        cout << static_cast<float>(score_ptr[x]) << ", " << threshold << endl;
        if( score_ptr[x] < threshold )
        {
            sprintf( name, "%s/I_new_%06d_old_%06d_%.3f.png", ResultsPath.c_str(), x, index, score_ptr[x] );
            imwrite(name,appended);
        }

        //imshow( "", appended );
        //imshow("matches", CorrespondenceImage);
        //waitKey(500);
        //fflush(stdout);
    }
    sprintf( name, "%s/matches.png", ResultsPath.c_str());
    imwrite(name,CorrespondenceImage);
}

void RunSeqSLAM(FileStorage fs)
{

    string TestPath = fs["FilePaths"]["TestPath"],
            QueryPath = fs["FilePaths"]["QueryPath"],
            ResultsPath = fs["SeqSLAM"]["ResultsPath"],
            CorrespondenceImageResults = fs["FilePaths"]["CorrespondenceImageResults"];

    float threshold = fs["SeqSLAM"]["Threshold"];
    int RWindow = fs["SeqSLAM"]["RWindow"];

    //VideoCapture newImages = loadDatasetVideo(QueryPath);
    //VideoCapture oldImages = loadDatasetVideo(TestPath);
    vector<Mat> newImages = loadDatasetFromVideo( QueryPath );
    vector<Mat> oldImages = loadDatasetFromVideo( TestPath );

    OpenSeqSLAM seq_slam;

    /* Preprocess the image set first */
    vector<Mat> preprocessed_new = seq_slam.preprocess( newImages );
    vector<Mat> preprocessed_old = seq_slam.preprocess( oldImages );

    seq_slam.RWindow = RWindow;
    /* Find the matches */
    const clock_t begin_time = clock();
    Mat matches = seq_slam.apply( preprocessed_new, preprocessed_old );
    //Mat matches = seq_slam.apply( newImages, oldImages );
    //cout << "SeqSLAM Total time: " << ((clock()- begin_time)/CLOCKS_PER_SECOND);
    Mat CorrespondenceImage = seq_slam.getCorrespondenceMatrix();

    double mi, ma;
    minMaxLoc(CorrespondenceImage,&mi,&ma);
    imwrite(CorrespondenceImageResults,255*(CorrespondenceImage-mi)/(ma-mi));

    //cout << "Show results" << endl;
    showLoopsDetections(matches, QueryPath, TestPath, CorrespondenceImage, ResultsPath, threshold);
}

int RunFABMapSeqSLAMOnlyMatches(FileStorage fs)
{
    string TestPath = fs["FilePaths"]["TestPath"],
            QueryPath = fs["FilePaths"]["QueryPath"],
            ResultsPath = fs["SchvarczSLAM"]["ResultsPath"],
            CorrespondenceImageResults= fs["FilePaths"]["CorrespondenceImageResults"];

    float threshold = fs["SchvarczSLAM"]["Threshold"],
            maxVar = fs["SchvarczSLAM"]["MaxVar"];
    int RWindow = fs["SchvarczSLAM"]["RWindow"],
            maxHalfWindowMeanShiftSize = fs["SchvarczSLAM"]["maxHalfWindowMeanShiftSize"];


    Ptr<FeatureDetector> detector = generateDetector(fs);
    if(!detector) {
        cerr << "Feature Detector error" << endl;
        return -1;
    }

    Ptr<DescriptorExtractor> extractor = generateExtractor(fs);
    if(!extractor) {
        cerr << "Feature Extractor error" << endl;
        return -1;
    }

    //vector<Mat> newImages = loadDatasetFromVideo( QueryPath );
    //vector<Mat> oldImages = loadDatasetFromVideo( TestPath );

    SchvaczSLAM schvarczSlam(detector,extractor);
    schvarczSlam.RWindow = RWindow;
    schvarczSlam.maxVar = maxVar;
    schvarczSlam.maxHalfWindowMeanShiftSize = maxHalfWindowMeanShiftSize;

    Mat CorrespondenceImage = imread(CorrespondenceImageResults);
    cvtColor(CorrespondenceImage,CorrespondenceImage,CV_RGB2GRAY);
    CorrespondenceImage.convertTo(CorrespondenceImage,CV_32F);
    double ma, mi;
    minMaxIdx(CorrespondenceImage,&mi,&ma);
    CorrespondenceImage = (CorrespondenceImage -mi)/(ma-mi);
    //CorrespondenceImage = -CorrespondenceImage+1;
    //CorrespondenceImage = CorrespondenceImage(Rect(0,55,65, CorrespondenceImage.rows-55));

    Mat matches = schvarczSlam.findMatches4(CorrespondenceImage);

    cout << matches.rows << " - " << matches.cols << endl;
    showLoopsDetections(matches, QueryPath, TestPath, CorrespondenceImage, ResultsPath, threshold);
    return 0;
}

int RunFABMapSeqSLAM(FileStorage fs)
{
    string TestPath = fs["FilePaths"]["TestPath"],
            QueryPath = fs["FilePaths"]["QueryPath"],
            ResultsPath = fs["SchvarczSLAM"]["ResultsPath"],
            SimilarityMatrixMode = fs["SchvarczSLAM"]["SimilarityMatrixMode"],
            CorrespondenceImageResults= fs["FilePaths"]["CorrespondenceImageResults"],
            vocabPath = fs["FilePaths"]["Vocabulary"],
            bowIDFWeightsPath = fs["FilePaths"]["IDFWeights"];

    float threshold = fs["SchvarczSLAM"]["Threshold"],
            maxVar = fs["SchvarczSLAM"]["MaxVar"];
    int RWindow = fs["SchvarczSLAM"]["RWindow"],
            maxHalfWindowMeanShiftSize = fs["SchvarczSLAM"]["maxHalfWindowMeanShiftSize"];

    Ptr<FeatureDetector> detector = generateDetector(fs);
    if(!detector) {
        cerr << "Feature Detector error" << endl;
        return -1;
    }

    Ptr<DescriptorExtractor> extractor = generateExtractor(fs);
    if(!extractor) {
        cerr << "Feature Extractor error" << endl;
        return -1;
    }

    //vector<Mat> newImages = loadDatasetFromVideo( QueryPath );
    //vector<Mat> oldImages = loadDatasetFromVideo( TestPath );
    VideoCapture newImages = loadDatasetVideo( QueryPath );
    VideoCapture oldImages = loadDatasetVideo( TestPath );

    SchvaczSLAM *schvarczSlam;
    if (SimilarityMatrixMode == "BOW")
    {
        int BOWType = generatorBOWType(fs["SchvarczSLAM"]["BOWType"]);

        //load vocabulary
        //cout << "Loading Vocabulary" << endl;
        fs.open(vocabPath, FileStorage::READ);
        Mat vocab;
        fs["Vocabulary"] >> vocab;
        if (vocab.empty()) {
            cerr << vocabPath << ": Vocabulary not found" << endl;
            return -1;
        }
        fs.release();


        //load vocabulary
        //cout << "Loading BOWIDFWeights" << endl;
        fs.open(bowIDFWeightsPath, FileStorage::READ);
        Mat bowIDFWeights;
        fs["BOWIDFWeights"] >> bowIDFWeights;
        if (bowIDFWeights.empty()) {
            cerr << bowIDFWeightsPath << ": BOWIDFWeights not found" << endl;
            return -1;
        }
        fs.release();


        schvarczSlam = new SchvaczSLAM(detector,extractor,vocab,bowIDFWeights.t());
        schvarczSlam->setBOWType(BOWType);
    }
    else if (SimilarityMatrixMode == "FeaturesMatching")
    {
        schvarczSlam = new SchvaczSLAM(detector,extractor);
    }
    schvarczSlam->RWindow = RWindow;
    schvarczSlam->maxVar = maxVar;
    schvarczSlam->maxHalfWindowMeanShiftSize = maxHalfWindowMeanShiftSize;

    const clock_t begin_time = clock();
    Mat matches = schvarczSlam->apply(newImages,oldImages);
    //cout << "SchvarczSLAM Total time: " << ((clock()- begin_time)/CLOCKS_PER_SECOND);
    Mat CorrespondenceImage = schvarczSlam->getCorrespondenceMatrix();

    double mi, ma;
    minMaxLoc(CorrespondenceImage,&mi,&ma);
    imwrite(CorrespondenceImageResults,255*(CorrespondenceImage-mi)/(ma-mi));

    showLoopsDetections(matches, QueryPath, TestPath, CorrespondenceImage, ResultsPath, threshold);
    return 0;
}

/*
The openFabMapcli accepts a YML settings file, an example of which is provided.
Modify options in the settings file for desired operation
*/
int RunFABMAP(FileStorage fs)
{
    Ptr<FeatureDetector> detector = generateDetector(fs);
    if(!detector) {
        cerr << "Feature Detector error" << endl;
        return -1;
    }

    Ptr<DescriptorExtractor> extractor = generateExtractor(fs);
    if(!extractor) {
        cerr << "Feature Extractor error" << endl;
        return -1;
    }

    //string extractorType = fs["FeatureOptions"]["ExtractorType"];
    //Ptr<DescriptorExtractor> extractor;
    //if(extractorType == "SIFT") {
    //	extractor = new SiftDescriptorExtractor();
    //} else if(extractorType == "SURF") {
    //	extractor = new SurfDescriptorExtractor(
    //		fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
    //		fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
    //		(int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
    //		(int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
    //} else {
    //	cerr << "Could not create Descriptor Extractor. Please specify "
    //		"extractor type in settings file" << endl;
    //	return -1;
    //}

    //run desired function
    int result = 0;
    string function = fs["Function"];
    if (function == "ShowFeatures") {
        result = showFeatures(
                    fs["FilePaths"]["TrainPath"],
                    detector);

    } else if (function == "GenerateVocabTrainData") {
        result = generateVocabTrainData(fs["FilePaths"]["TrainPath"],
                                        fs["FilePaths"]["TrainFeatDesc"],
                                        detector, extractor);

    } else if (function == "GenerateVocabTrainDataWafer") {
        result = generateVocabTrainDataPatch(fs["FilePaths"]["TrainPath"],
                                        fs["FilePaths"]["TrainFeatDesc"],
                                        detector, extractor);

    } else if (function == "TrainVocabulary") {
        result = trainVocabulary(fs["VocabTrainOptions"]["BOWTrainerType"],
                                 fs["FilePaths"]["Vocabulary"],
                                 fs["FilePaths"]["TrainFeatDesc"],
                                 fs["VocabTrainOptions"]["ClusterSize"]);

    } else if (function == "GenerateFABMAPTrainData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TrainPath"],
                                       fs["FilePaths"]["TrainImagDesc"],
                                       fs["FilePaths"]["Vocabulary"], detector, extractor,
                                       fs["BOWOptions"]["MinWords"]);

    }else if (function == "GenerateFABMAPTrainDataWafer") {
        result = generateBOWImageDescsWafer(fs["FilePaths"]["TrainPath"],
                                       fs["FilePaths"]["TrainImagDesc"],
                                       fs["FilePaths"]["Vocabulary"], detector, extractor,
                                       fs["BOWOptions"]["MinWords"]);

    }
    else if (function == "GenerateIDFTrainData") {
        result = generateBOWIDFWeights(fs["FilePaths"]["TrainImagDesc"],
                                       fs["FilePaths"]["IDFWeights"]);

    } else if (function == "TrainChowLiuTree") {
        result = trainChowLiuTree(fs["FilePaths"]["ChowLiuTree"],
                                  fs["FilePaths"]["TrainImagDesc"],
                                  fs["ChowLiuOptions"]["LowerInfoBound"]);

    } else if (function == "GenerateFABMAPTestData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TestPath"],
                                       fs["FilePaths"]["TestImageDesc"],
                                       fs["FilePaths"]["Vocabulary"], detector, extractor,
                                       fs["BOWOptions"]["MinWords"]);

    } else if (function == "RunOpenFABMAP") {
        string placeAddOption = fs["FabMapPlaceAddition"];
        bool addNewOnly = (placeAddOption == "NewMaximumOnly");
        of2c::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
                result = openFABMAP(fs["FilePaths"]["TestImageDesc"], fabmap,
                        fs["FilePaths"]["Vocabulary"],
                        fs["FilePaths"]["FabMapResults"], addNewOnly);
        }

    }  else if (function == "RunOpenFABMAPByTranning") {
        of2c::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
            result = openFABMAP(fs["FilePaths"]["TestPath"],
                                fs["FilePaths"]["QueryPath"], fabmap,
                                fs["FilePaths"]["Vocabulary"],
                                fs["FilePaths"]["CorrespondenceImageResults"],
                                fs["openFabMapOptions"]["ResultsPath"], detector, extractor);
        }

    } else {
        cerr << "Incorrect Function Type" << endl;
        result = -1;
    }

    cout << "openFABMAP done" << endl;
    cin.sync(); cin.ignore();

    fs.release();

    return result;
}


/*
The openFabMapcli accepts a YML settings file, an example of which is provided.
Modify options in the settings file for desired operation
*/
int RunFABMAPFull(FileStorage fs)
{
    Ptr<FeatureDetector> detector = generateDetector(fs);
    if(!detector) {
        cerr << "Feature Detector error" << endl;
        return -1;
    }

    Ptr<DescriptorExtractor> extractor = generateExtractor(fs);
    if(!extractor) {
        cerr << "Feature Extractor error" << endl;
        return -1;
    }

    //string extractorType = fs["FeatureOptions"]["ExtractorType"];
    //Ptr<DescriptorExtractor> extractor;
    //if(extractorType == "SIFT") {
    //	extractor = new SiftDescriptorExtractor();
    //} else if(extractorType == "SURF") {
    //	extractor = new SurfDescriptorExtractor(
    //		fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
    //		fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
    //		(int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
    //		(int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
    //} else {
    //	cerr << "Could not create Descriptor Extractor. Please specify "
    //		"extractor type in settings file" << endl;
    //	return -1;
    //}

    //run desired function
    int result = 0;
    string function = fs["Function"];
    result += generateVocabTrainData(fs["FilePaths"]["TrainPath"],
                                    fs["FilePaths"]["TrainFeatDesc"],
                                    detector, extractor);

    result += trainVocabulary(fs["VocabTrainOptions"]["BOWTrainerType"],
                              fs["FilePaths"]["Vocabulary"],
                             fs["FilePaths"]["TrainFeatDesc"],
                             fs["VocabTrainOptions"]["ClusterSize"]);

    result += generateBOWImageDescs(fs["FilePaths"]["TrainPath"],
                                   fs["FilePaths"]["TrainImagDesc"],
                                   fs["FilePaths"]["Vocabulary"], detector, extractor,
                                   fs["BOWOptions"]["MinWords"]);

    result += generateBOWIDFWeights(fs["FilePaths"]["TrainImagDesc"],
                                   fs["FilePaths"]["IDFWeights"]);

    result += trainChowLiuTree(fs["FilePaths"]["ChowLiuTree"],
                              fs["FilePaths"]["TrainImagDesc"],
                              fs["ChowLiuOptions"]["LowerInfoBound"]);

    result += generateBOWImageDescs(fs["FilePaths"]["TestPath"],
                                   fs["FilePaths"]["TestImageDesc"],
                                   fs["FilePaths"]["Vocabulary"], detector, extractor,
                                   fs["BOWOptions"]["MinWords"]);

    if (function == "RunOpenFABMAP") {
        string placeAddOption = fs["FabMapPlaceAddition"];
        bool addNewOnly = (placeAddOption == "NewMaximumOnly");
        of2c::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
                result += openFABMAP(fs["FilePaths"]["TestImageDesc"], fabmap,
                        fs["FilePaths"]["Vocabulary"],
                        fs["FilePaths"]["FabMapResults"], addNewOnly);
        }

    }  else if (function == "RunOpenFABMAPByTranning") {
        of2c::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
            result += openFABMAP(fs["FilePaths"]["TestPath"],
                                fs["FilePaths"]["QueryPath"], fabmap,
                                fs["FilePaths"]["Vocabulary"],
                                fs["FilePaths"]["CorrespondenceImageResults"],
                                fs["openFabMapOptions"]["ResultsPath"], detector, extractor);
        }

    } else {
        cerr << "Incorrect Function Type" << endl;
        result = -1;
    }

    cout << "openFABMAP done" << endl;
    //cin.sync(); cin.ignore();

    fs.release();

    return result;
}


/*
displays the usage message
*/
int help(void)
{
    cout << "Usage: FABSeqSLAM -s settingsfile" << endl;
    return 0;
}

void testLSE1()
{
    cout << "TesteLSE1" << endl;
    Mat pts(2,5,CV_32F);
    for (int x=0;x<pts.cols;x++)
        pts.at<float>(0,x) = float(x);
    pts.at<float>(1,  0 ) = 2 ;
    pts.at<float>(1,  1 ) = 0 ;
    pts.at<float>(1,  2 ) = 0;
    pts.at<float>(1,  3 ) = 2;
    pts.at<float>(1,  4 ) = 6;
    pts = pts.t();

    LSE lse;
    Mat params = lse.model(pts);
    cout << "Params: " << pts << endl;
    cout << "Params: " << params << endl;
}

void testLSE2()
{
    cout << "TesteLSE2" << endl;
    Mat pts(2,100,CV_32F);
    for (int x=0;x<pts.cols;x++)
        pts.at<float>(0,x) = float(x);
    pts.at<float>(1,  0 ) = -445.267302865 ;
    pts.at<float>(1,  1 ) = -100.003592336 ;
    pts.at<float>(1,  2 ) = -297.050598923 ;
    pts.at<float>(1,  3 ) = -111.414221236 ;
    pts.at<float>(1,  4 ) = 941.825076237 ;
    pts.at<float>(1,  5 ) = -321.34525499 ;
    pts.at<float>(1,  6 ) = 438.482637526 ;
    pts.at<float>(1,  7 ) = -702.706451548 ;
    pts.at<float>(1,  8 ) = -341.783632207 ;
    pts.at<float>(1,  9 ) = 612.997082795 ;
    pts.at<float>(1,  10 ) = 441.313244946 ;
    pts.at<float>(1,  11 ) = 532.955583181 ;
    pts.at<float>(1,  12 ) = -442.040207288 ;
    pts.at<float>(1,  13 ) = 771.211974348 ;
    pts.at<float>(1,  14 ) = 648.917795481 ;
    pts.at<float>(1,  15 ) = 336.764539538 ;
    pts.at<float>(1,  16 ) = 767.525485044 ;
    pts.at<float>(1,  17 ) = -894.718013378 ;
    pts.at<float>(1,  18 ) = 421.009523594 ;
    pts.at<float>(1,  19 ) = 867.951457584 ;
    pts.at<float>(1,  20 ) = -11.5334207505 ;
    pts.at<float>(1,  21 ) = 1237.74961439 ;
    pts.at<float>(1,  22 ) = -93.0346884437 ;
    pts.at<float>(1,  23 ) = 775.103644994 ;
    pts.at<float>(1,  24 ) = 1075.50394509 ;
    pts.at<float>(1,  25 ) = -148.249684919 ;
    pts.at<float>(1,  26 ) = 1013.56552909 ;
    pts.at<float>(1,  27 ) = 639.356105195 ;
    pts.at<float>(1,  28 ) = 1449.69550958 ;
    pts.at<float>(1,  29 ) = 780.910674168 ;
    pts.at<float>(1,  30 ) = 826.37799663 ;
    pts.at<float>(1,  31 ) = 1319.91181648 ;
    pts.at<float>(1,  32 ) = 1183.11322 ;
    pts.at<float>(1,  33 ) = 1575.05565735 ;
    pts.at<float>(1,  34 ) = 371.917868799 ;
    pts.at<float>(1,  35 ) = 1869.38014792 ;
    pts.at<float>(1,  36 ) = 961.872044281 ;
    pts.at<float>(1,  37 ) = 1599.23556361 ;
    pts.at<float>(1,  38 ) = 1083.17328461 ;
    pts.at<float>(1,  39 ) = 2193.58228314 ;
    pts.at<float>(1,  40 ) = 1953.14855119 ;
    pts.at<float>(1,  41 ) = 2103.00076541 ;
    pts.at<float>(1,  42 ) = 2550.27032003 ;
    pts.at<float>(1,  43 ) = 1392.59207363 ;
    pts.at<float>(1,  44 ) = 2344.14293519 ;
    pts.at<float>(1,  45 ) = 2306.54716779 ;
    pts.at<float>(1,  46 ) = 2235.929142 ;
    pts.at<float>(1,  47 ) = 2564.01764847 ;
    pts.at<float>(1,  48 ) = 2325.22137544 ;
    pts.at<float>(1,  49 ) = 2691.19966432 ;
    pts.at<float>(1,  50 ) = 3227.4397964 ;
    pts.at<float>(1,  51 ) = 2226.89151462 ;
    pts.at<float>(1,  52 ) = 2510.83469783 ;
    pts.at<float>(1,  53 ) = 2747.43731226 ;
    pts.at<float>(1,  54 ) = 2098.93980048 ;
    pts.at<float>(1,  55 ) = 3123.97023786 ;
    pts.at<float>(1,  56 ) = 3257.28757952 ;
    pts.at<float>(1,  57 ) = 3150.87113505 ;
    pts.at<float>(1,  58 ) = 3578.51273722 ;
    pts.at<float>(1,  59 ) = 3085.86862447 ;
    pts.at<float>(1,  60 ) = 4321.62325808 ;
    pts.at<float>(1,  61 ) = 4491.41985395 ;
    pts.at<float>(1,  62 ) = 3792.53850562 ;
    pts.at<float>(1,  63 ) = 3360.3285873 ;
    pts.at<float>(1,  64 ) = 4065.37726169 ;
    pts.at<float>(1,  65 ) = 3899.47192729 ;
    pts.at<float>(1,  66 ) = 5130.21869209 ;
    pts.at<float>(1,  67 ) = 4710.64870619 ;
    pts.at<float>(1,  68 ) = 4356.36506042 ;
    pts.at<float>(1,  69 ) = 5223.03245095 ;
    pts.at<float>(1,  70 ) = 4549.51439295 ;
    pts.at<float>(1,  71 ) = 5390.51094568 ;
    pts.at<float>(1,  72 ) = 6026.65848251 ;
    pts.at<float>(1,  73 ) = 4539.22421737 ;
    pts.at<float>(1,  74 ) = 5018.27577085 ;
    pts.at<float>(1,  75 ) = 5487.32862174 ;
    pts.at<float>(1,  76 ) = 6545.73485707 ;
    pts.at<float>(1,  77 ) = 5832.87366622 ;
    pts.at<float>(1,  78 ) = 5672.22241403 ;
    pts.at<float>(1,  79 ) = 6344.11705429 ;
    pts.at<float>(1,  80 ) = 6891.56831752 ;
    pts.at<float>(1,  81 ) = 6268.97131779 ;
    pts.at<float>(1,  82 ) = 6678.228422 ;
    pts.at<float>(1,  83 ) = 7250.13475912 ;
    pts.at<float>(1,  84 ) = 7278.91216909 ;
    pts.at<float>(1,  85 ) = 6873.24595323 ;
    pts.at<float>(1,  86 ) = 7636.97699997 ;
    pts.at<float>(1,  87 ) = 7659.83934359 ;
    pts.at<float>(1,  88 ) = 7068.84868161 ;
    pts.at<float>(1,  89 ) = 8554.74555516 ;
    pts.at<float>(1,  90 ) = 8113.68680936 ;
    pts.at<float>(1,  91 ) = 8479.56964217 ;
    pts.at<float>(1,  92 ) = 8856.79691885 ;
    pts.at<float>(1,  93 ) = 9406.90128541 ;
    pts.at<float>(1,  94 ) = 9368.47796499 ;
    pts.at<float>(1,  95 ) = 8902.93120658 ;
    pts.at<float>(1,  96 ) = 9492.40077676 ;
    pts.at<float>(1,  97 ) = 9648.48946518 ;
    pts.at<float>(1,  98 ) = 8582.89427901 ;
    pts.at<float>(1,  99 ) = 9509.88587441 ;
    pts = pts.t();

    LSE lse;
    Mat params = lse.model(pts);
    cout << "Params: " << params << endl;
}

void testLSE3()
{
    cout << "TesteLSE3" << endl;
    Mat pts(2,100,CV_32F);
    for (int x=0;x<pts.cols;x++)
        pts.at<float>(0,x) = float(x);
    pts.at<float>(1,  0 ) = -445.267302865 ;
    pts.at<float>(1,  1 ) = -100.003592336 ;
    pts.at<float>(1,  2 ) = -297.050598923 ;
    pts.at<float>(1,  3 ) = -111.414221236 ;
    pts.at<float>(1,  4 ) = 941.825076237 ;
    pts.at<float>(1,  5 ) = -321.34525499 ;
    pts.at<float>(1,  6 ) = 438.482637526 ;
    pts.at<float>(1,  7 ) = -702.706451548 ;
    pts.at<float>(1,  8 ) = -341.783632207 ;
    pts.at<float>(1,  9 ) = 612.997082795 ;
    pts.at<float>(1,  10 ) = 441.313244946 ;
    pts.at<float>(1,  11 ) = 532.955583181 ;
    pts.at<float>(1,  12 ) = -442.040207288 ;
    pts.at<float>(1,  13 ) = 771.211974348 ;
    pts.at<float>(1,  14 ) = 648.917795481 ;
    pts.at<float>(1,  15 ) = 336.764539538 ;
    pts.at<float>(1,  16 ) = 767.525485044 ;
    pts.at<float>(1,  17 ) = -894.718013378 ;
    pts.at<float>(1,  18 ) = 421.009523594 ;
    pts.at<float>(1,  19 ) = 867.951457584 ;
    pts.at<float>(1,  20 ) = -11.5334207505 ;
    pts.at<float>(1,  21 ) = 1237.74961439 ;
    pts.at<float>(1,  22 ) = -93.0346884437 ;
    pts.at<float>(1,  23 ) = 775.103644994 ;
    pts.at<float>(1,  24 ) = 1075.50394509 ;
    pts.at<float>(1,  25 ) = -148.249684919 ;
    pts.at<float>(1,  26 ) = 1013.56552909 ;
    pts.at<float>(1,  27 ) = 639.356105195 ;
    pts.at<float>(1,  28 ) = 1449.69550958 ;
    pts.at<float>(1,  29 ) = 780.910674168 ;
    pts.at<float>(1,  30 ) = 826.37799663 ;
    pts.at<float>(1,  31 ) = 1319.91181648 ;
    pts.at<float>(1,  32 ) = 1183.11322 ;
    pts.at<float>(1,  33 ) = 1575.05565735 ;
    pts.at<float>(1,  34 ) = 371.917868799 ;
    pts.at<float>(1,  35 ) = 1869.38014792 ;
    pts.at<float>(1,  36 ) = 961.872044281 ;
    pts.at<float>(1,  37 ) = 1599.23556361 ;
    pts.at<float>(1,  38 ) = 1083.17328461 ;
    pts.at<float>(1,  39 ) = 2193.58228314 ;
    pts.at<float>(1,  40 ) = 1953.14855119 ;
    pts.at<float>(1,  41 ) = 2103.00076541 ;
    pts.at<float>(1,  42 ) = 2550.27032003 ;
    pts.at<float>(1,  43 ) = 1392.59207363 ;
    pts.at<float>(1,  44 ) = 2344.14293519 ;
    pts.at<float>(1,  45 ) = 2306.54716779 ;
    pts.at<float>(1,  46 ) = 2235.929142 ;
    pts.at<float>(1,  47 ) = 2564.01764847 ;
    pts.at<float>(1,  48 ) = 2325.22137544 ;
    pts.at<float>(1,  49 ) = 2691.19966432 ;
    pts.at<float>(1,  50 ) = 3227.4397964 ;
    pts.at<float>(1,  51 ) = 2226.89151462 ;
    pts.at<float>(1,  52 ) = 2510.83469783 ;
    pts.at<float>(1,  53 ) = 2747.43731226 ;
    pts.at<float>(1,  54 ) = 2098.93980048 ;
    pts.at<float>(1,  55 ) = 3123.97023786 ;
    pts.at<float>(1,  56 ) = 3257.28757952 ;
    pts.at<float>(1,  57 ) = 3150.87113505 ;
    pts.at<float>(1,  58 ) = 3578.51273722 ;
    pts.at<float>(1,  59 ) = 3085.86862447 ;
    pts.at<float>(1,  60 ) = 4321.62325808 ;
    pts.at<float>(1,  61 ) = 4491.41985395 ;
    pts.at<float>(1,  62 ) = 3792.53850562 ;
    pts.at<float>(1,  63 ) = 3360.3285873 ;
    pts.at<float>(1,  64 ) = 4065.37726169 ;
    pts.at<float>(1,  65 ) = 3899.47192729 ;
    pts.at<float>(1,  66 ) = 5130.21869209 ;
    pts.at<float>(1,  67 ) = 4710.64870619 ;
    pts.at<float>(1,  68 ) = 4356.36506042 ;
    pts.at<float>(1,  69 ) = 5223.03245095 ;
    pts.at<float>(1,  70 ) = 4549.51439295 ;
    pts.at<float>(1,  71 ) = 5390.51094568 ;
    pts.at<float>(1,  72 ) = 6026.65848251 ;
    pts.at<float>(1,  73 ) = 4539.22421737 ;
    pts.at<float>(1,  74 ) = 5018.27577085 ;
    pts.at<float>(1,  75 ) = 5487.32862174 ;
    pts.at<float>(1,  76 ) = 6545.73485707 ;
    pts.at<float>(1,  77 ) = 5832.87366622 ;
    pts.at<float>(1,  78 ) = 5672.22241403 ;
    pts.at<float>(1,  79 ) = 6344.11705429 ;
    pts.at<float>(1,  80 ) = 6891.56831752 ;
    pts.at<float>(1,  81 ) = 6268.97131779 ;
    pts.at<float>(1,  82 ) = 6678.228422 ;
    pts.at<float>(1,  83 ) = 7250.13475912 ;
    pts.at<float>(1,  84 ) = 7278.91216909 ;
    pts.at<float>(1,  85 ) = 6873.24595323 ;
    pts.at<float>(1,  86 ) = 7636.97699997 ;
    pts.at<float>(1,  87 ) = 7659.83934359 ;
    pts.at<float>(1,  88 ) = 7068.84868161 ;
    pts.at<float>(1,  89 ) = 8554.74555516 ;
    pts.at<float>(1,  90 ) = 8113.68680936 ;
    pts.at<float>(1,  91 ) = 8479.56964217 ;
    pts.at<float>(1,  92 ) = 8856.79691885 ;
    pts.at<float>(1,  93 ) = 9406.90128541 ;
    pts.at<float>(1,  94 ) = 9368.47796499 ;
    pts.at<float>(1,  95 ) = 8902.93120658 ;
    pts.at<float>(1,  96 ) = 9492.40077676 ;
    pts.at<float>(1,  97 ) = 9648.48946518 ;
    pts.at<float>(1,  98 ) = 8582.89427901 ;
    pts.at<float>(1,  99 ) = 9509.88587441 ;
    pts = pts.t();

    LSE lse;
    Mat params = lse.model(pts);
    Mat inliers = lse.fit(pts, params, 500);
    cout << "Inliers: " << inliers.rows << endl;
}

void testLSE4()
{
    cout << "TesteLSE4" << endl;
    Mat pts(2,100,CV_32F);
    for (int x=0;x<pts.cols;x++)
        pts.at<float>(0,x) = float(x);
    pts.at<float>(1,  0 ) = -445.267302865 ;
    pts.at<float>(1,  1 ) = -100.003592336 ;
    pts.at<float>(1,  2 ) = -297.050598923 ;
    pts.at<float>(1,  3 ) = -111.414221236 ;
    pts.at<float>(1,  4 ) = 941.825076237 ;
    pts.at<float>(1,  5 ) = -321.34525499 ;
    pts.at<float>(1,  6 ) = 438.482637526 ;
    pts.at<float>(1,  7 ) = -702.706451548 ;
    pts.at<float>(1,  8 ) = -341.783632207 ;
    pts.at<float>(1,  9 ) = 612.997082795 ;
    pts.at<float>(1,  10 ) = 441.313244946 ;
    pts.at<float>(1,  11 ) = 532.955583181 ;
    pts.at<float>(1,  12 ) = -442.040207288 ;
    pts.at<float>(1,  13 ) = 771.211974348 ;
    pts.at<float>(1,  14 ) = 648.917795481 ;
    pts.at<float>(1,  15 ) = 336.764539538 ;
    pts.at<float>(1,  16 ) = 767.525485044 ;
    pts.at<float>(1,  17 ) = -894.718013378 ;
    pts.at<float>(1,  18 ) = 421.009523594 ;
    pts.at<float>(1,  19 ) = 867.951457584 ;
    pts.at<float>(1,  20 ) = -11.5334207505 ;
    pts.at<float>(1,  21 ) = 1237.74961439 ;
    pts.at<float>(1,  22 ) = -93.0346884437 ;
    pts.at<float>(1,  23 ) = 775.103644994 ;
    pts.at<float>(1,  24 ) = 1075.50394509 ;
    pts.at<float>(1,  25 ) = -148.249684919 ;
    pts.at<float>(1,  26 ) = 1013.56552909 ;
    pts.at<float>(1,  27 ) = 639.356105195 ;
    pts.at<float>(1,  28 ) = 1449.69550958 ;
    pts.at<float>(1,  29 ) = 780.910674168 ;
    pts.at<float>(1,  30 ) = 826.37799663 ;
    pts.at<float>(1,  31 ) = 1319.91181648 ;
    pts.at<float>(1,  32 ) = 1183.11322 ;
    pts.at<float>(1,  33 ) = 1575.05565735 ;
    pts.at<float>(1,  34 ) = 371.917868799 ;
    pts.at<float>(1,  35 ) = 1869.38014792 ;
    pts.at<float>(1,  36 ) = 961.872044281 ;
    pts.at<float>(1,  37 ) = 1599.23556361 ;
    pts.at<float>(1,  38 ) = 1083.17328461 ;
    pts.at<float>(1,  39 ) = 2193.58228314 ;
    pts.at<float>(1,  40 ) = 1953.14855119 ;
    pts.at<float>(1,  41 ) = 2103.00076541 ;
    pts.at<float>(1,  42 ) = 2550.27032003 ;
    pts.at<float>(1,  43 ) = 1392.59207363 ;
    pts.at<float>(1,  44 ) = 2344.14293519 ;
    pts.at<float>(1,  45 ) = 2306.54716779 ;
    pts.at<float>(1,  46 ) = 2235.929142 ;
    pts.at<float>(1,  47 ) = 2564.01764847 ;
    pts.at<float>(1,  48 ) = 2325.22137544 ;
    pts.at<float>(1,  49 ) = 2691.19966432 ;
    pts.at<float>(1,  50 ) = 3227.4397964 ;
    pts.at<float>(1,  51 ) = 2226.89151462 ;
    pts.at<float>(1,  52 ) = 2510.83469783 ;
    pts.at<float>(1,  53 ) = 2747.43731226 ;
    pts.at<float>(1,  54 ) = 2098.93980048 ;
    pts.at<float>(1,  55 ) = 3123.97023786 ;
    pts.at<float>(1,  56 ) = 3257.28757952 ;
    pts.at<float>(1,  57 ) = 3150.87113505 ;
    pts.at<float>(1,  58 ) = 3578.51273722 ;
    pts.at<float>(1,  59 ) = 3085.86862447 ;
    pts.at<float>(1,  60 ) = 4321.62325808 ;
    pts.at<float>(1,  61 ) = 4491.41985395 ;
    pts.at<float>(1,  62 ) = 3792.53850562 ;
    pts.at<float>(1,  63 ) = 3360.3285873 ;
    pts.at<float>(1,  64 ) = 4065.37726169 ;
    pts.at<float>(1,  65 ) = 3899.47192729 ;
    pts.at<float>(1,  66 ) = 5130.21869209 ;
    pts.at<float>(1,  67 ) = 4710.64870619 ;
    pts.at<float>(1,  68 ) = 4356.36506042 ;
    pts.at<float>(1,  69 ) = 5223.03245095 ;
    pts.at<float>(1,  70 ) = 4549.51439295 ;
    pts.at<float>(1,  71 ) = 5390.51094568 ;
    pts.at<float>(1,  72 ) = 6026.65848251 ;
    pts.at<float>(1,  73 ) = 4539.22421737 ;
    pts.at<float>(1,  74 ) = 5018.27577085 ;
    pts.at<float>(1,  75 ) = 5487.32862174 ;
    pts.at<float>(1,  76 ) = 6545.73485707 ;
    pts.at<float>(1,  77 ) = 5832.87366622 ;
    pts.at<float>(1,  78 ) = 5672.22241403 ;
    pts.at<float>(1,  79 ) = 6344.11705429 ;
    pts.at<float>(1,  80 ) = 6891.56831752 ;
    pts.at<float>(1,  81 ) = 6268.97131779 ;
    pts.at<float>(1,  82 ) = 6678.228422 ;
    pts.at<float>(1,  83 ) = 7250.13475912 ;
    pts.at<float>(1,  84 ) = 7278.91216909 ;
    pts.at<float>(1,  85 ) = 6873.24595323 ;
    pts.at<float>(1,  86 ) = 7636.97699997 ;
    pts.at<float>(1,  87 ) = 7659.83934359 ;
    pts.at<float>(1,  88 ) = 7068.84868161 ;
    pts.at<float>(1,  89 ) = 8554.74555516 ;
    pts.at<float>(1,  90 ) = 8113.68680936 ;
    pts.at<float>(1,  91 ) = 8479.56964217 ;
    pts.at<float>(1,  92 ) = 8856.79691885 ;
    pts.at<float>(1,  93 ) = 9406.90128541 ;
    pts.at<float>(1,  94 ) = 9368.47796499 ;
    pts.at<float>(1,  95 ) = 8902.93120658 ;
    pts.at<float>(1,  96 ) = 9492.40077676 ;
    pts.at<float>(1,  97 ) = 9648.48946518 ;
    pts.at<float>(1,  98 ) = 8582.89427901 ;
    pts.at<float>(1,  99 ) = 9509.88587441 ;
    pts = pts.t();

    LSE lse;
    Mat params = lse.model(pts);
    Mat inliers = lse.fit(pts, params, 500);
    Mat newPts = lse.compute(pts,params);
    cout << "Compute: " << newPts << endl;
}

void draw(Mat img, Vector< Point > pts)
{
    Mat drawImg; img.copyTo(drawImg);
    cvtColor(drawImg, drawImg, CV_GRAY2RGB);
    for (int i=0; i<pts.size(); i++)
        circle(drawImg,pts[i],1,Scalar(255,0,0),-1);
    imshow("Line", drawImg);
    //waitKey();
}

Mat vectorToMat(Vector < Point > pts)
{
    Mat ret(pts.size(), 2, CV_32F, 0.0);

    for (int i=0; i < pts.size(); i++)
    {
        ret.at<float>(i,0) = pts[i].x;
        ret.at<float>(i,1) = pts[i].y;
    }

    return ret;
}

Vector < Point > matToVector(Mat pts)
{
    Vector < Point > ret;

    for (int i=0; i <pts.rows; i++)
        ret.push_back(Point(pts.at<float>(i,0), pts.at<float>(i,1)));

    return ret;
}

float lineRank(Mat img, Mat pts)
{
    float mean = 0;
    for(int i=0; i<pts.rows; i++)
        mean += img.at<float>(pts.at<float>(i,1),pts.at<float>(i,0));
    mean /= pts.rows;

    return mean;
}

void testMeanShift1()
{
    cout << "TestMeanShift 1: MeanShift" << endl;
    Mat img;
    cvtColor(imread("results.bmp"),img,CV_RGB2GRAY);
    img.convertTo(img,CV_32F);
    imshow("original",img);
    cout << img.rows << " x " << img.cols << endl;
    Rect roi(10,140,40,40);
    Mat imgRect = img(roi);
    imshow("rect", imgRect);

    double ma,mi;
    minMaxLoc(imgRect, &mi, &ma);
    imgRect = (imgRect-mi)/(ma-mi);
    imgRect = -imgRect + 1;
    imshow("rectN", imgRect);

    double maxHalfWindowMeanShiftSize = 5;

    Vector < Point > line ;
    for(int i=0;i<=30;i++)
    {
        line.push_back(Point(i+maxHalfWindowMeanShiftSize, i+maxHalfWindowMeanShiftSize));
    }

    draw(imgRect,line);

    cout << "Rodando meanshift... " << endl;
    SchvaczSLAM SSLAM;
    SSLAM.findMatch4(imgRect, line);
    cout << "done!" << endl;
    draw(imgRect,line);

    for(int i=0;i<30;i++)
    {
        cout << line[i].x << " - " << line[i].y << endl;
    }

    cout << "LineRank: " << lineRank(imgRect,vectorToMat(line)) << endl;

}

void testMeanShift2()
{
    cout << "TestMeanShift 2: LSE" << endl;
    Mat img;
    cvtColor(imread("results.bmp"),img,CV_RGB2GRAY);
    img.convertTo(img,CV_32F);

    Rect roi(10,140,40,40);
    Mat imgRect = img(roi);

    double ma,mi;
    minMaxLoc(imgRect, &mi, &ma);
    imgRect = (imgRect-mi)/(ma-mi);
    imgRect = -imgRect + 1;

    double maxHalfWindowMeanShiftSize = 5;

    Vector < Point > line ;
    for(int i=0;i<=30;i++)
    {
        line.push_back(Point(i+maxHalfWindowMeanShiftSize, i+maxHalfWindowMeanShiftSize));
    }

    SchvaczSLAM SSLAM;
    SSLAM.findMatch4(imgRect, line);

    //Start LSE
    Mat pts = vectorToMat(line);
    LSE lse;
    Mat params = lse.model(pts);
    pts = lse.compute(pts, params);

    line = matToVector(pts);
    Mat imgRect3;
    cvtColor(imgRect, imgRect3, CV_GRAY2RGB);

    for (int i=0; i<line.size()-1; i++)
        cv::line(imgRect3, line[i], line[i+1], Scalar(255,0,0));

    imshow("Final!!!", imgRect3);

    cout << "LineRank: " << lineRank(imgRect,pts) << endl;
}
void testMeanShift3()
{
    cout << "TestMeanShift 3: Ransac" << endl;
    Mat img;
    cvtColor(imread("results.bmp"),img,CV_RGB2GRAY);
    img.convertTo(img,CV_32F);

    Rect roi(10,140,40,40);
    Mat imgRect = img(roi);

    double ma,mi;
    minMaxLoc(imgRect, &mi, &ma);
    imgRect = (imgRect-mi)/(ma-mi);
    imgRect = -imgRect + 1;

    double maxHalfWindowMeanShiftSize = 5;

    Vector < Point > line ;
    for(int i=0;i<=30;i++)
    {
        line.push_back(Point(i+maxHalfWindowMeanShiftSize, i+maxHalfWindowMeanShiftSize));
    }

    SchvaczSLAM SSLAM;
    SSLAM.findMatch4(imgRect, line);

    Mat pts = vectorToMat(line);
    LSE lse;

    Ransac ransac(lse);
    pts = ransac.compute(pts);
    line = matToVector(pts);
    Mat imgRect3;
    cvtColor(imgRect, imgRect3, CV_GRAY2RGB);
    for (int i=0; i<line.size()-1; i++)
        cv::line(imgRect3, line[i], line[i+1], Scalar(255,0,0));

    imshow("Final Ransac!!!", imgRect3);
    cout << "LineRank: " << lineRank(imgRect,pts) << endl;
    waitKey();
}

void testRange()
{
    float maxHalfWindowMeanShiftSize =5, xMax = 29;
    float x[] = {2,5,10,23,24,27};
    for(int i=0; i<6;i++)
        cout << x[i] << ": " << MIN(maxHalfWindowMeanShiftSize - MAX(maxHalfWindowMeanShiftSize - x[i],0),
                                    maxHalfWindowMeanShiftSize - MAX(maxHalfWindowMeanShiftSize + x[i] - xMax,0)) << endl;
}

void testSuite()
{
//    testLSE1();
//    testLSE2();
//    testLSE3();
//    testLSE4();
    testMeanShift1();
//    testMeanShift2();
//    testMeanShift3();
    testRange();
}

int main(int argc, const char * argv[])
{

    cout.setf(ios_base::fixed);
    cout.precision(2);

    //load the settings file
    string settfilename;
    if (argc == 1) {
        //assume settings in working directory
        settfilename = "settings.yml";
    } else if (argc == 3) {
        if(string(argv[1]) != "-s") {
            //incorrect option
            return help();
        } else {
            //settings provided as argument
            settfilename = string(argv[2]);
        }
    } else {
        //incorrect arguments
        return help();
    }

    FileStorage fs;
    fs.open(settfilename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Could not open settings file: " << settfilename <<
                     endl;
        return -1;
    }

    string SLAMType = fs["SLAM"]["Type"];
    cout << "SLAMType: " << SLAMType << endl;
    if ( SLAMType == "FABSeqSLAM" )
        RunFABMapSeqSLAM(fs);
    else if ( SLAMType == "SeqSLAM" )
        RunSeqSLAM(fs);
    else if ( SLAMType == "FABMap" )
        RunFABMAP(fs);
    else if ( SLAMType == "FABMapFull" )
        RunFABMAPFull(fs);
    else if ( SLAMType == "FABSeqSLAMOnlyMatches" )
        RunFABMapSeqSLAMOnlyMatches(fs);
    else if ( SLAMType == "TestSuite" )
        testSuite();

    return 0;
}

