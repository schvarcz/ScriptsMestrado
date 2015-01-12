#ifndef FABSEQSLAM_H
#define FABSEQSLAM_H

#include <OpenSeqSLAM.h>

using namespace std;
using namespace cv;

class FABSeqSLAM : public OpenSeqSLAM
{
public:
  FABSeqSLAM();
  Mat apply(vector<Mat> &set_1, vector<Mat> &set_2);
};

#endif // FABSEQSLAM_H
