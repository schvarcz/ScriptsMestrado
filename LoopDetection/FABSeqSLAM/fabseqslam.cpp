#include "fabseqslam.h"

FABSeqSLAM::FABSeqSLAM(): OpenSeqSLAM()
{
}

Mat FABSeqSLAM::apply(Mat img)
{
  cout << "FABSeqSlam" << endl;

  enhanced = Mat(img.size(),img.type(), Scalar(255)) - img;

  imshow("enhanced", enhanced);

  return findMatches( enhanced );
}
