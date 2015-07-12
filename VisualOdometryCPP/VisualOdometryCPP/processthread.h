#ifndef PROCESSTHREAD_H
#define PROCESSTHREAD_H

#include <QtCore>
#include <QThread>
#include <QDebug>
#include <QDir>
#include <iostream>
#include <fstream>
#include <vector>
#include <viso/viso_mono.h>
#include <png++/png.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ProcessThread: public QThread
{
    Q_OBJECT
public:
    ProcessThread(VisualOdometryMono::parameters param, QString odometryImagesPath, QString filePattern, int step);
    ProcessThread(VisualOdometryMono::parameters param, QString odometryImagesPath, QString odometryOutputFile);

protected:
    void run();
    void gerarDadosComLibviso(QString defaultPath);
    void gerarDadosCV(QString defaultPath, QString savePath);
    void drawFeaturesCorrespondence(Mat& img, vector<IMatcher::p_match> fts, const Scalar& colorLine, const Scalar& colorPoint);
    void drawFeatures(Mat& img, vector<KeyPoint> fts, const Scalar& colorPoint);

private:
    int step;
    VisualOdometryMono::parameters param;
    QString odometryImagesPath, filePattern, odometryOutputFile, featuresImagesOutputPath;
};

#endif // PROCESSTHREAD_H
