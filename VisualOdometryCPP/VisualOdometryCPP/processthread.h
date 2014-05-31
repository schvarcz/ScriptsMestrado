#ifndef PROCESSTHREAD_H
#define PROCESSTHREAD_H

#include <QtCore>
#include <QThread>
#include <QDebug>
#include <QDir>
#include <iostream>
#include <fstream>
#include <vector>
#include <viso_mono.h>
#include <png++/png.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ProcessThread: public QThread
{
    Q_OBJECT
public:
    ProcessThread(VisualOdometryMono::parameters param, QString dir, QString filePattern, int step, ofstream *positions, ofstream *features);

protected:
    void run();
    void gerarDadosComLibviso(QString defaultPath, QString savePath, int step);
    void gerarDadosCV(QString defaultPath, QString savePath, int step);

private:
    int step;
    VisualOdometryMono::parameters param;
    QString dir, filePattern;
    ofstream *positions, *features;
};

#endif // PROCESSTHREAD_H
