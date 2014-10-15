#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    ui->pushButton->setEnabled(false);
    // set most important visual odometry parameters
    // for a full parameter list, look at: viso_stereo.h
    VisualOdometryMono::parameters param;

    if (!ui->CBRobo->currentText().compare("Libviso"))
    {
        qDebug() << "Libviso";
        // calibration parameters for sequence 2010_03_09_drive_0019
        param.calib.f  = 645.24; // focal length in pixels
        param.calib.cu = 635.96; // principal point (u-coordinate) in pixels
        param.calib.cv = 194.13; // principal point (v-coordinate) in pixels
    }
    else if (!ui->CBRobo->currentText().compare("NAO"))
    {
        qDebug() << "NAO";
        // calibration parameters for sequence nao
        param.calib.f  = 545.790894; // focal length in pixels
        param.calib.cu = 320.824327; // principal point (u-coordinate) in pixels
        param.calib.cv = 244.557631; // principal point (v-coordinate) in pixels
    }
    else if (!ui->CBRobo->currentText().compare("Ardrone"))
    {
        qDebug() << "Ardrone";
        // calibration parameters for sequence drone
        param.calib.f  = 567.102880658; // focal length in pixels
        param.calib.cu = 315.76714448; // principal point (u-coordinate) in pixels
        param.calib.cv = 164.95995903; // principal point (v-coordinate) in pixels
    }
    else if (!ui->CBRobo->currentText().compare("Ardrone Half"))
    {
        qDebug() << "Ardrone half";
        // calibration parameters for sequence drone
        param.calib.f  = 206.533682058; // focal length in pixels
        param.calib.cu = 140.81438384; // principal point (u-coordinate) in pixels
        param.calib.cv = 78.91016624; // principal point (v-coordinate) in pixels
    }
    else if (!ui->CBRobo->currentText().compare("Rodrigo"))
    {
        qDebug() << "Rodrigo"; //Copia do ardrone
        // calibration parameters for sequence drone
        param.calib.f  = 565.2659195; // focal length in pixels
        param.calib.cu = 320.583306; // principal point (u-coordinate) in pixels
        param.calib.cv = 164.804138; // principal point (v-coordinate) in pixels
    }
    else if (!ui->CBRobo->currentText().compare("Motox"))
    {
        qDebug() << "MotoX";
        // calibration parameters for sequence drone
        param.calib.f  = 2.31584916e+03; // focal length in pixels
        param.calib.cu = 1.00664394e+03; // principal point (u-coordinate) in pixels
        param.calib.cv = 5.04108086e+02; // principal point (v-coordinate) in pixels
    }
    else if (!ui->CBRobo->currentText().compare("Motox Half"))
    {
        qDebug() << "MotoX half size";
        param.calib.f  = 1.31842535e+03; // focal length in pixels
        param.calib.cu = 5.11004989e+02; // principal point (u-coordinate) in pixels
        param.calib.cv = 2.35035536e+02; // principal point (v-coordinate) in pixels
    }
    // sequence directory
    QString dir = ui->LEDir->text(), filePattern = ui->LENamePattern->text();
    int step = ui->SBFrameStep->value();

//    posicoes.open((dir+"/posicoes_%1.csv").arg(step).toAscii());
//    features.open((dir+"/features_%1.csv").arg(step).toAscii());


    pt = new ProcessThread(param, dir, filePattern, step, &posicoes, &features);
    connect(pt,SIGNAL(finished()),this,SLOT(reactiveButton()));
    pt->start();
}

void MainWindow::reactiveButton()
{
    qDebug() << "Desliga tudo!";
//    posicoes.close();
//    features.close();
    ui->pushButton->setEnabled(true);
}
