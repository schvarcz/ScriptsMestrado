#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QDebug>
#include <QDir>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <processthread.h>

#include <viso_mono.h>
#include <png++/png.hpp>

using namespace std;
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    
private slots:
    void on_pushButton_clicked();
    void reactiveButton();

private:
    Ui::MainWindow *ui;
    ProcessThread *pt;

    ofstream posicoes, features;

};

#endif // MAINWINDOW_H
