#-------------------------------------------------
#
# Project created by QtCreator 2014-03-24T20:36:21
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VisualOdometryCPP
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    viso_stereo.cpp \
    viso_mono.cpp \
    viso.cpp \
    triangle.cpp \
    reconstruction.cpp \
    matrix.cpp \
    matcher.cpp \
    filter.cpp \
    processthread.cpp \
    matchercv.cpp \
    imatcher.cpp \
    robot.cpp

HEADERS  += mainwindow.h \
    viso_stereo.h \
    viso_mono.h \
    viso.h \
    triangle.h \
    timer.h \
    reconstruction.h \
    matrix.h \
    matcher.h \
    filter.h \
    processthread.h \
    matchercv.h \
    imatcher.h \
    robot.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -msse3

LIBS += -lpng12

INCLUDEPATH += -I/usr/local/include/opencv -I/usr/local/include
LIBS += -L/usr/local/lib/ -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lrt -lpthread -lm -ldl

OTHER_FILES += \
    VisualOdometryCPP.pro.user \
    settings.yml
