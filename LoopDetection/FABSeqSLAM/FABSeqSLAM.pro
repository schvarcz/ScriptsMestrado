
QT       += core gui

TEMPLATE = app

SOURCES += main.cpp \
    OpenSeqSLAM.cpp \
    fabseqslam.cpp \
    FabMap.cpp \
    ChowLiuTree.cpp \
    BOWMSCTrainer.cpp \
    SchvarczSLAM.cpp \
    lse.cpp \
    ransac.cpp

HEADERS += \
    OpenSeqSLAM.h \
    fabseqslam.h \
    openfabmap.hpp \
    SchvarczSLAM.h \
    lse.h \
    ransac.h



INCLUDEPATH += -I/usr/local/include/opencv -I/usr/local/include
LIBS += -L/usr/local/lib/ -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab -lrt -lpthread -lm -ldl

OTHER_FILES += \
    settings.yml
