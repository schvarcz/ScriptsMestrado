#include "processthread.h"

ProcessThread::ProcessThread(VisualOdometryMono::parameters param, QString dir, QString filePattern, int step): QThread()
{
    this->param = param;
    this->odometryImagesPath = dir;
    this->filePattern = filePattern;
}

ProcessThread::ProcessThread(VisualOdometryMono::parameters param, QString odometryImagesPath, QString odometryOutputFile): QThread()
{
    this->param = param;
    this->odometryImagesPath = odometryImagesPath;
    this->odometryOutputFile = odometryOutputFile;
    filePattern = QString("I_%06d.png");
}

void ProcessThread::run()
{
    QString sPath = ""; //savePath + "/step_eq_match_sift_" + QString::number(1);

    //QDir diretory;
    //diretory.mkpath(sPath);

    gerarDadosCV(this->odometryImagesPath,sPath);
    //gerarDadosComLibviso(this->odometryImagesPath);
}

void ProcessThread::gerarDadosComLibviso(QString defaultPath)
{
        ofstream *positions = new ofstream();

        positions->open(this->odometryOutputFile.toStdString().c_str());

        // init visual odometry
        VisualOdometryMono viso(param);

        // current pose (this matrix transforms a point from the current
        // frame's camera coordinates to the first frame's camera coordinates)
        Matrix rot = Matrix(6,1);
        Matrix pose = Matrix::eye(4);

        QDir diretory(defaultPath);
        QStringList filtro;
        filtro << "*.png";
        int nFrames = diretory.entryList(filtro).count();
        qDebug() << nFrames;

        bool replace = false;
        // loop through all frames i=0:372
        for (int32_t i=0; i<nFrames; i++) {

            // input file names
            char base_name[256];
            sprintf(base_name, filePattern.toStdString().c_str(),i);
            QString img_file_name  = defaultPath + "/" + base_name;

            // catch image read/write errors here
            try {

                // load left and right input image
                png::image< png::gray_pixel > img(img_file_name.toStdString().c_str());

                // image dimensions
                int32_t width  = img.get_width();
                int32_t height = img.get_height();

                // convert input images to uint8_t buffer
                uint8_t* img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
                int32_t k=0;
                for (int32_t v=0; v<height; v++) {
                    for (int32_t u=0; u<width; u++) {
                        img_data[k]  = img.get_pixel(u,v);
                        k++;
                    }
                }

                // status
                qDebug() << "Processing: Frame: " << i << "/" << nFrames;

                // compute visual odometry
                int32_t dims[] = {width,height,width};
                if (viso.process(img_data,dims,replace  && i!=1)) {
                    replace = false;
                    qDebug() << "Processing: Frame: " << i;

                    // on success, update current pose
                    pose = pose * Matrix::inv(viso.getMotion());
                    rot = rot + viso.getMotionVector();

                    // output some statistics
                    double num_matches = viso.getNumberOfMatches();
                    double num_inliers = viso.getNumberOfInliers();
                    qDebug() << ", Matches: " << num_matches;
                    qDebug() << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;

                    //qDebug() << pose << endl << endl;

                } else {
                    replace = true;
                    qDebug() << " ... failed!" << endl;
                }

                qDebug() << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << ", " << rot.val[0][0] << "," << rot.val[1][0] << "," << rot.val[2][0] << endl;
                *positions << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << ", " << rot.val[0][0] << "," << rot.val[1][0] << "," << rot.val[2][0] << endl;

                // release uint8_t buffers
                free(img_data);

                // catch image read errors here
            } catch (...) {
                qDebug() << "ERROR: Couldn't read input files!";
            }
        }
        positions->close();
}


void ProcessThread::gerarDadosCV(QString defaultPath, QString savePath)
{
    ofstream *positions = new ofstream();

    positions->open(this->odometryOutputFile.toStdString().c_str());

    // init visual odometry
    VisualOdometryMono viso(param);

    // current pose (this matrix transforms a point from the current
    // frame's camera coordinates to the first frame's camera coordinates)
    Matrix rot = Matrix(6,1);
    Matrix pose = Matrix::eye(4);

    QDir diretory(defaultPath);
    QStringList filtro;
    filtro << "*.png";

    int nFrames = diretory.count();

    bool replace = false;

    // loop through all frames
    for (int i=0; i<nFrames; i++) {

        // input file names
        char base_name[256];
        sprintf(base_name, filePattern.toStdString().c_str(),i);
        QString img_file_name  = defaultPath + "/" + base_name;

        // catch image read/write errors here
        try {

            // load left and right input image
            Mat img = imread(img_file_name.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);

            // status
            qDebug() << "Processing: Frame: " << i << "/" << nFrames;

            // compute visual odometry
            if (viso.process(img,replace && i!=1 && i!=2)) {

                // on success, update current pose
                Matrix estimated = viso.getMotion();
                Matrix estimatedMotionVector = viso.getMotionVector();

                if (
                        10 < sqrt(pow(estimatedMotionVector.val[3][0],2) + pow(estimatedMotionVector.val[4][0],2)  + pow(estimatedMotionVector.val[5][0],2))
//                        && 0.2 > sqrt(pow(estimatedMotionVector.val[3][0],2) + pow(estimatedMotionVector.val[4][0],2)  + pow(estimatedMotionVector.val[5][0],2))
                        )
                {
                    qDebug() << "Ignorar movimento longo" << endl;
                    replace = true;
                    continue;
                }

                pose = pose * Matrix::inv(estimated);
                rot = rot + estimatedMotionVector;
                replace = false;

                // output some statistics
                double num_matches = viso.getNumberOfMatches();
                double num_inliers = viso.getNumberOfInliers();
                qDebug() << ", Matches: " << num_matches;
                qDebug() << ", Inliers: " << 100.0*num_inliers/num_matches << "%" << endl;

                //qDebug() << pose << endl << endl;

            } else {
                replace = true;
                // output some statistics
                double num_matches = viso.getNumberOfMatches();
                qDebug() << ", Matches: " << num_matches;
                qDebug() << " ... failed!" << endl;
            }

            vector<IMatcher::p_match> fts = viso.getFeatures();
            cvtColor(img,img,CV_GRAY2RGB);
            this->drawFeatures(img,viso.getFeaturesCV(), Scalar(255,0, 0));
            this->drawFeaturesCorrespondence(img,fts, Scalar(0,255,0), Scalar(255,0,255));
            this->drawFeaturesCorrespondence(img,viso.getInliers(), Scalar(0, 0, 255), Scalar(0,255,255));
            //imshow("features",img);

            if (savePath != "")
            {
                QString fileName = QString("/I1_%0.png").arg(QString::number(i/step),6, QChar('0'));
                imwrite((savePath+fileName).toStdString(),img);
            }
            cout << "Current pose: " << endl << pose << endl;
            *positions << img_file_name.toStdString() << "," << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << ", " << rot.val[0][0] << "," << rot.val[1][0] << "," << rot.val[2][0]  << endl;

            // release uint8_t buffers
            img.release();

            // catch image read errors here
        } catch (...) {
            qDebug() << "ERROR: Couldn't read input files!";
            break;
        }
    }
    positions->close();
}

void ProcessThread::drawFeaturesCorrespondence(Mat& img, vector<IMatcher::p_match> fts, const Scalar& colorLine, const Scalar& colorPoint)
{
    for (vector<IMatcher::p_match>::iterator it = fts.begin(); it!=fts.end(); it++)
    {
        Point ptc(it->u1c,it->v1c);
        Point ptp(it->u1p,it->v1p);
        line(img,ptc,ptp,colorLine);
        circle(img,ptc,3,colorPoint, -1);
    }
}


void ProcessThread::drawFeatures(Mat& img, vector<KeyPoint> fts, const Scalar& colorPoint)
{
    for (vector<KeyPoint>::iterator it = fts.begin(); it!=fts.end(); it++)
    {
        circle(img,it->pt,3,colorPoint, -1);
    }
}
