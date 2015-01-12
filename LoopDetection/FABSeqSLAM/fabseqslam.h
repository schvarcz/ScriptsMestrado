#ifndef FABSEQSLAM_H
#define FABSEQSLAM_H

#include <OpenSeqSLAM.h>

using namespace std;
using namespace cv;

class FABSeqSLAM : public OpenSeqSLAM
{
public:
  FABSeqSLAM();
  Mat apply(Mat img);
};

#endif // FABSEQSLAM_H
