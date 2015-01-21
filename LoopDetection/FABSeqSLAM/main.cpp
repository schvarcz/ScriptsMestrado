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

#ifdef OPENCV2P4
#include <opencv2/nonfree/nonfree.hpp>
#endif

#include "openfabmap.hpp"
#include "OpenSeqSLAM.h"
#include "fabseqslam.h"
#include "SchvarczSLAM.h"

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
int trainVocabulary(string vocabPath,
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
        if(waitKey(5) == 27) {
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
int trainVocabulary(string vocabPath,
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
    of2c::BOWMSCTrainer trainer(clusterRadius);
    trainer.add(vocabTrainData);
    Mat vocab = trainer.cluster();

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
        cout << "\r"<<  indexWord << "° Word " << 100.0*indexWord/bowImageDesc.cols << "%                  ";
        fflush(stdout);
        float idf = log(bowImageDesc.rows/occurrence);
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
Mat openFABMAP(Mat queryImageDescs,
               Mat testImageDescs,
               of2c::FabMap *fabmap)
{

    //running openFABMAP
    cout << "Running openFABMAP" << endl;
    vector<of2c::IMatch> matches;
    vector<of2c::IMatch>::iterator l;

    Mat confusion_mat(queryImageDescs.rows, testImageDescs.rows, CV_64FC1);
    confusion_mat = Scalar(0); // init to 0's

    //automatically comparing a whole dataset
    fabmap->compare(queryImageDescs, testImageDescs, matches);
    cout.precision(3);
    for(l = matches.begin(); l != matches.end(); l++) {
        confusion_mat.at<double>(l->queryIdx, l->imgIdx) = l->match;
    }
    cout.precision(0);
    return confusion_mat;
}


int openFABMAP(string TestPath,
               string QueryPath,
               of2c::FabMap *fabmap,
               string vocabPath,
               string CorrespondenceImageResults,
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

    cout << "Generating BOW for test images: " << TestPath << endl;
    Mat BOWTest = generateBOWImageDescs(TestPath, vocab, detector, extractor);
    cout << "Generating BOW for query images: " << QueryPath << endl;
    Mat BOWQuery = generateBOWImageDescs(QueryPath, vocab, detector, extractor);

    if(fabmap) {
        Mat result = openFABMAP(BOWQuery,
                                BOWTest,
                                fabmap);

        cout << "Saving results: " << CorrespondenceImageResults << endl;
        double ma, mi;
        minMaxLoc(result,&mi,&ma);
        result = 255*(result - mi)/ma;
        imwrite(CorrespondenceImageResults,result);

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

    } else {
        cerr << "Could not create Descriptor Extractor. Please specify "
                     "extractor type in settings file" << endl;
    }

    return extractor;

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

    cout << "Loading video: " << path << endl;
    Mat frame;

    while (movie.read(frame)) {

        cout << "\r " << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                                movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%         ";
        fflush(stdout);

        Mat save;
        frame.copyTo(save);
        images.push_back( save );
    }
    cout << "Done" << endl;

    return images;
}

void showLoopsDetections(Mat matches, vector<Mat> newImages, vector<Mat> oldImages, Mat CorrespondenceImage, string ResultsPath,float threshold = 0.99)
{
    cvtColor(CorrespondenceImage,CorrespondenceImage,CV_GRAY2RGB);
    CvFont font = cvFontQt("Helvetica", 20.0, CV_RGB(255, 0, 0) );
    namedWindow("");
    moveWindow("", 0, 0);

    char temp[100], name[100];

    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);

    Mat appended( newImages[0].rows,newImages[0].cols*2, newImages[0].type(), Scalar(0) );

    for( uint x = 0; x < newImages.size(); x++ ) {

        cout << "\r Image " << x << "/" << newImages.size() << " (" << 100.0*float(x)/newImages.size()<< "%)      ";
        fflush(stdout);

        int index = static_cast<int>(index_ptr[x]);

        //cout << index << endl;
        /* Append the images together */
        appended.setTo(Scalar(0));

        newImages[x].copyTo( Mat(appended, Rect(0, 0, appended.cols/2, appended.rows) ));

        if( score_ptr[x] < threshold )
            oldImages[index].copyTo( Mat(appended, Rect(appended.cols/2, 0, appended.cols/2, appended.rows) ));

        /* The lower the score, the lower the differences between images */
        if( score_ptr[x] < threshold )
            sprintf( temp, "Old image [%03d]", index );
        else
            sprintf( temp, "Old image [None]" );

        addText( appended, temp, Point( appended.cols/2 + 20, 20 ), font );

        sprintf( temp, "New image [%03d]", x );
        addText( appended, temp, Point( 10, 20 ), font );


        cout << score_ptr[x];
        if( score_ptr[x] < threshold )
        {
            sprintf( name, "%s/matches/I_new_%06d_old_%06d.png", ResultsPath.c_str(), x,index );
            imwrite(name,appended);
        }

        circle(CorrespondenceImage,Point(index,x),1,Scalar(255,0,0),-1);
        imshow( "", appended );
        imshow("matches", CorrespondenceImage);
        waitKey(500);
    }
}

void RunSeqSLAM(FileStorage fs)
{

    string TestPath = fs["FilePaths"]["TestPath"],
            QueryPath = fs["FilePaths"]["QueryPath"],
            ResultsPath = fs["SeqSLAM"]["ResultsPath"],
            CorrespondenceImageResults = fs["FilePaths"]["CorrespondenceImageResults"];

    vector<Mat> newImages = loadDatasetFromVideo( QueryPath );
    vector<Mat> oldImages = loadDatasetFromVideo( TestPath );

    OpenSeqSLAM seq_slam;

    /* Preprocess the image set first */
    vector<Mat> preprocessed_new = seq_slam.preprocess( newImages );
    vector<Mat> preprocessed_old = seq_slam.preprocess( oldImages );

    /* Find the matches */
    Mat matches = seq_slam.apply( preprocessed_new, preprocessed_old );
    Mat CorrespondenceImage = seq_slam.getCorrespondenceMatrix();

    imwrite(CorrespondenceImageResults,CorrespondenceImage);

    showLoopsDetections(matches, newImages, oldImages, CorrespondenceImage, ResultsPath, 0.99);
}

int RunFABMapSeqSLAMOnlyMatches(FileStorage fs)
{
    string TestPath = fs["FilePaths"]["TestPath"],
            QueryPath = fs["FilePaths"]["QueryPath"],
            ResultsPath = fs["SeqSLAM"]["ResultsPath"],
            CorrespondenceImageResults= fs["FilePaths"]["CorrespondenceImageResults"];

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

    vector<Mat> newImages = loadDatasetFromVideo( QueryPath );
    vector<Mat> oldImages = loadDatasetFromVideo( TestPath );
    cout << newImages.size() << " - " << oldImages.size();

    SchvaczSLAM schvarczSlam(detector,extractor);

    Mat CorrespondenceImage = imread(CorrespondenceImageResults);
    cvtColor(CorrespondenceImage,CorrespondenceImage,CV_RGB2GRAY);
    //CorrespondenceImage = CorrespondenceImage(Rect(0,55,65, CorrespondenceImage.rows-55));

    Mat matches = schvarczSlam.findMatches2(CorrespondenceImage);

    cout << matches.rows << " - " << matches.cols << endl;
    showLoopsDetections(matches, newImages, oldImages, CorrespondenceImage, ResultsPath, 1500);
    return 0;
}

int RunFABMapSeqSLAM(FileStorage fs)
{
    string TestPath = fs["FilePaths"]["TestPath"],
            QueryPath = fs["FilePaths"]["QueryPath"],
            ResultsPath = fs["SeqSLAM"]["ResultsPath"],
            CorrespondenceImageResults= fs["FilePaths"]["CorrespondenceImageResults"],
            vocabPath = fs["FilePaths"]["Vocabulary"],
            bowIDFWeightsPath = fs["FilePaths"]["IDFWeights"];

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

    //    of2c::FabMap *fabmap = generateFABMAPInstance(fs);

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


    //load vocabulary
    cout << "Loading BOWIDFWeights" << endl;
    fs.open(bowIDFWeightsPath, FileStorage::READ);
    Mat bowIDFWeights;
    fs["BOWIDFWeights"] >> bowIDFWeights;
    if (bowIDFWeights.empty()) {
        cerr << bowIDFWeightsPath << ": BOWIDFWeights not found" << endl;
        return -1;
    }
    fs.release();

    vector<Mat> newImages = loadDatasetFromVideo( QueryPath );
    vector<Mat> oldImages = loadDatasetFromVideo( TestPath );

    SchvaczSLAM schvarczSlam(detector,extractor,vocab,bowIDFWeights.t());
    Mat matches = schvarczSlam.apply(newImages,oldImages);
    Mat CorrespondenceImage = schvarczSlam.getCorrespondenceMatrix();

    imwrite(CorrespondenceImageResults,255*CorrespondenceImage);

    cout << matches.rows << " - " << matches.cols << endl;
    showLoopsDetections(matches, newImages, oldImages, CorrespondenceImage, ResultsPath, 1500);
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

    } else if (function == "TrainVocabulary") {
        result = trainVocabulary(fs["FilePaths"]["Vocabulary"],
                                 fs["FilePaths"]["TrainFeatDesc"],
                                 fs["VocabTrainOptions"]["ClusterSize"]);

    } else if (function == "GenerateFABMAPTrainData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TrainPath"],
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
                                fs["FilePaths"]["CorrespondenceImageResults"], detector, extractor);
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
displays the usage message
*/
int help(void)
{
    cout << "Usage: FABSeqSLAM -s settingsfile" << endl;
    return 0;
}

int main(int argc, const char * argv[])
{

    cout.setf(ios_base::fixed);
    cout.precision(0);

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
    else if ( SLAMType == "FABSeqSLAMOnlyMatches" )
        RunFABMapSeqSLAMOnlyMatches(fs);

    return 0;
}

