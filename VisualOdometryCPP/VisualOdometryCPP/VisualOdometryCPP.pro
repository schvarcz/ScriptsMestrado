#-------------------------------------------------
#
# Project created by QtCreator 2014-03-24T20:36:21
#
#-------------------------------------------------

QT       += core gui

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
    imatcher.cpp

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
    imatcher.h

FORMS    += mainwindow.ui

QMAKE_CXXFLAGS += -msse3

LIBS += -lpng12 -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_video

OTHER_FILES += \
    VisualOdometryCPP.pro.user
