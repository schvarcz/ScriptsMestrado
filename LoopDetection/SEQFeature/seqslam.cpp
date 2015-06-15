#include "seqslam.h"

SeqSLAM::SeqSLAM()
{
}

void SeqSLAM::operator ()(VideoCapture dataset1, VideoCapture dataset2)
{
    int size1 = dataset1.get(CV_CAP_PROP_FRAME_COUNT);
    int size2 = dataset2.get(CV_CAP_PROP_FRAME_COUNT);

    similarityMatrix.create(size2,size1,CV_32F);
    similarityMatrix.setTo(Scalar(0));

    Mat frame1, frame2;
    while(dataset2.read(frame2))
    {
        dataset1.set(CV_CAP_PROP_POS_FRAMES,0.0);
        while(dataset1.read(frame1))
        {
        }
    }
}
