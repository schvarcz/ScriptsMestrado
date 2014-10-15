#include "processthread.h"

ProcessThread::ProcessThread(VisualOdometryMono::parameters param, QString dir, QString filePattern, int step, ofstream *positions, ofstream *features): QThread()
{
    this->param = param;
    this->dir = dir;
    this->filePattern = filePattern;
    this->positions = positions;
    this->features = features;
    this->step = step;
}

void ProcessThread::run()
{
    QString defaultPath = "/home/schvarcz/Dissertacao/datasets/";
    QString savePath = "/home/schvarcz/Dissertacao/OdometriaVisual/features/";

    QStringList paths;
    paths
//            << "2010_03_09_drive_0019"
//            << "drone/20140318_132620"
//            << "drone/20140318_132620_gray"
//            << "drone/20140318_133931"
//            << "drone/20140318_133931_gray"
//            << "drone/20140327_135316_gray"
//            << "drone/20140328_102444_gray"
//            << "drone/video_20140327_135316"
            << "drone/video_20140328_102444"
//            << "motox/VID_20140617_162058756_GRAY"
//            << "motox/VID_20140617_162058756_GRAY_ESCOLHA"
//            << "motox/VID_20140617_162058756_GRAY_ESCOLHA_GRAMA"
//            << "motox/VID_20140617_162058756_GRAY_equalized"
//            << "motox/VID_20140617_162058756_GRAY_equalized_small"
//            << "motox/VID_20140617_162058756_GRAY_CURVA"
//            << "motox/VID_20140617_162058756_GRAY_small_CURVA"
//            << "motox/VID_20140617_162058756_GRAY_equalized_small_CURVA"
//            << "motox/VID_20140617_162058756_GRAY_equalized_small_GRAMA_ESCOLHA"
//            << "motox/VID_20140617_162058756_GRAY_equalized_small_ESCOLHA"
//            << "motox/VID_20140617_162058756_GRAY_equalized_small_ESCOLHA2"
//            << "motox/VID_20140617_163505406_GRAY"
//            << "car_simulation/carro_stereo_2014-06-12-02-14-53_2"
//            << "nao/nao2"
//            << "nao/nao2_gray"
//            << "nao/nao2_rect"
//            << "nao/nao2_rect_escolha"
//            << "nao/naooo_2014-03-10-17-48-35"
//            << "nao/naooo_2014-03-10-17-48-35_gray"
             ;

    //    int steps[] = {25,20,15,10,5,3,2,1};
    int steps[] = {1};
//    int steps[] = {15};
    for(int s = 0;s<1;s++)
    {
        for(QStringList::iterator path = paths.begin(); path != paths.end(); path++)
        {
            QDir diretory;
            QString sPath = savePath+*path + "/step_eq_match_sift_" + QString::number(steps[s]);
            diretory.mkpath(sPath);
            gerarDadosCV(defaultPath+*path,sPath ,steps[s]);
//            gerarDadosComLibviso(defaultPath+paths[0],sPath,steps[s]);
        }
    }
}

void ProcessThread::gerarDadosComLibviso(QString defaultPath, QString savePath, int step)
{
        features = new ofstream();
        positions = new ofstream();

        features->open((savePath+"/features_%1.csv").arg(step).toAscii());
        positions->open((savePath+"/posicoes_%1.csv").arg(step).toAscii());

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
        for (int32_t i=0; i<nFrames; i+=step) {

            // input file names
            char base_name[256];
            sprintf(base_name, filePattern.toAscii(),i);
            QString img_file_name  = defaultPath + "/" + base_name;
            *features << "imagem " << i << endl;

            // catch image read/write errors here
            try {

                // load left and right input image
                png::image< png::gray_pixel > img(img_file_name.toAscii());

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

                vector<IMatcher::p_match> fts = viso.getFeatures();
                for (vector<IMatcher::p_match>::iterator it = fts.begin(); it!=fts.end(); it++)
                {
                    *features << it->u1p << "," << it->v1p << " ; " << it->u1c << ", " << it->v1c << endl;
                }
                qDebug() << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << ", " << rot.val[0][0] << "," << rot.val[1][0] << "," << rot.val[2][0] << endl;
                *positions << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << ", " << rot.val[0][0] << "," << rot.val[1][0] << "," << rot.val[2][0] << endl;
//                *positions << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << endl;

                // release uint8_t buffers
                free(img_data);

                // catch image read errors here
            } catch (...) {
                qDebug() << "ERROR: Couldn't read input files!";
            }
        }
        features->close();
        positions->close();
}


void ProcessThread::gerarDadosCV(QString defaultPath, QString savePath, int step)
{
    features = new ofstream();
    positions = new ofstream();

    features->open((savePath+"/features_%1.csv").arg(step).toAscii());
    positions->open((savePath+"/posicoes_%1.csv").arg(step).toAscii());

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
    for (int i=0; i<nFrames; i+=step) {

        // input file names
        char base_name[256];
        sprintf(base_name, filePattern.toAscii(),i);
        QString img_file_name  = defaultPath + "/" + base_name;
        *features << "imagem " << i << endl;

        // catch image read/write errors here
        try {

            // load left and right input image
            Mat img = imread(img_file_name.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);

            // status
            qDebug() << "Processing: Frame: " << i << "/" << nFrames;

            // compute visual odometry
            if (viso.process(img,replace && i!=step && i!=2*step)) {

                // on success, update current pose
                Matrix estimated = viso.getMotion();
                Matrix estimatedMotionVector = viso.getMotionVector();

                if (
                        10 < sqrt(pow(estimatedMotionVector.val[3][0],2) + pow(estimatedMotionVector.val[4][0],2)  + pow(estimatedMotionVector.val[5][0],2))
//                        && 0.2 > sqrt(pow(estimatedMotionVector.val[3][0],2) + pow(estimatedMotionVector.val[4][0],2)  + pow(estimatedMotionVector.val[5][0],2))
                        )
                {
                    cout << "Ignorar movimento longo" << endl;
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
                qDebug() << ", Inliers: " << 100.0*num_inliers/num_matches << " %" << ", Current pose: " << endl;

                //qDebug() << pose << endl << endl;

            } else {
                replace = true;
                // output some statistics
                double num_matches = viso.getNumberOfMatches();
                qDebug() << ", Matches: " << num_matches;
                qDebug() << " ... failed!" << endl;
            }

            vector<IMatcher::p_match> fts = viso.getFeatures();
            for (vector<IMatcher::p_match>::iterator it = fts.begin(); it!=fts.end(); it++)
            {
                *features << it->u1p << "," << it->v1p << " ; " << it->u1c << ", " << it->v1c << endl;
            }
            cvtColor(img,img,CV_GRAY2RGB);
            this->drawFeatures(img,viso.getFeaturesCV(), Scalar(255,0, 0));
            this->drawFeaturesCorrespondence(img,fts, Scalar(0,255,0), Scalar(255,0,255));
            this->drawFeaturesCorrespondence(img,viso.getInliers(), Scalar(0, 0, 255), Scalar(0,255,255));
//            imshow("features",img);

            QString fileName = QString("/I1_%0.png").arg(QString::number(i/step),6, QChar('0'));
            imwrite((savePath+fileName).toStdString(),img);
            qDebug() << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3];
//            *positions << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << endl;
            *positions << pose.val[0][3] << ", " << pose.val[1][3] << ", " << pose.val[2][3] << ", " << rot.val[0][0] << "," << rot.val[1][0] << "," << rot.val[2][0] << "," << img_file_name.toStdString() << endl;

            // release uint8_t buffers
            img.release();

            // catch image read errors here
        } catch (...) {
            qDebug() << "ERROR: Couldn't read input files!";
            break;
        }
    }
    features->close();
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
