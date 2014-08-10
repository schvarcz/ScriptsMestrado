#include <QtGui/QApplication>
#include "mainwindow.h"

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

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
//    return testeMatchCV();
}
