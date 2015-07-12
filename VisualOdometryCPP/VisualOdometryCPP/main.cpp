#include <QApplication>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <viso/viso_mono.h>
#include <mainwindow.h>

using namespace std;

/*
displays the usage message
*/
int help(void)
{
    cout << "Usage: VisualOdometryCPP -s settingsfile" << endl;
    return 0;
}
int testeMatchCV()
{
    Mat img1 = imread("I1_000009.png"), img2 = imread("I1_000011.png"), final = imread("I1_000011.png");
    MatcherCV matcher("SIFT","SIFT");
    matcher.pushBack(img1);
    matcher.matchFeatures();
    matcher.pushBack(img2);
    matcher.matchFeatures();

    vector<IMatcher::p_match> matches;
    matches = matcher.getMatches();


    for(vector<IMatcher::p_match>::iterator it = matches.begin();it != matches.end(); it++)
    {
        line(final,Point((*it).u1p,(*it).v1p),Point((*it).u1c,(*it).v1c),Scalar(0,255,0));
        circle(final,Point((*it).u1c,(*it).v1c),3,Scalar(255,0,0),-1);
    }

    imshow("Final", final);
    waitKey();
    return 1;
}

int qtInterface(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
//    return testeMatchCV();
}



int main(int argc, char * argv[])
{
//    qtInterface(argc,argv);

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

    string OdometryImagesPath = fs["FilePaths"]["OdometryImagesPath"],
            OdometryOutputFile = fs["FilePaths"]["OdometryOutputFile"],
            FeaturesImagesOutputPath = fs["FilePaths"]["FeaturesImagesOutputPath"];
    float f = fs["CameraMatrix"]["f"],
            cu = fs["CameraMatrix"]["cu"],
            cv = fs["CameraMatrix"]["cv"];

    // set most important visual odometry parameters
    // for a full parameter list, look at: viso_stereo.h
    VisualOdometryMono::parameters param;

    param.calib.f  = f; // focal length in pixels
    param.calib.cu = cu; // principal point (u-coordinate) in pixels
    param.calib.cv = cv; // principal point (v-coordinate) in pixels

    cout << OdometryImagesPath << " " << f << " " << cu << " " << cv << endl;
    ProcessThread *pt = new ProcessThread(param,
                                          QString::fromStdString(OdometryImagesPath),
                                          QString::fromStdString(OdometryOutputFile));
    pt->start();

    pt->wait();

    return 0;
}
