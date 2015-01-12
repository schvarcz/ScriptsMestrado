#include "fabseqslam.h"

FABSeqSLAM::FABSeqSLAM(): OpenSeqSLAM()
{
}


Mat FABSeqSLAM::apply(vector<Mat> &set_1, vector<Mat> &set_2)
{
  cout << "FABSeqSlam" << endl;
  Mat img = imread("/home/schvarcz/Dissertacao/src/LoopDetection/FABMap-build-desktop-Qt_4_8_1_in_PATH__System__Release/results.bmp");

  enhanced = Mat(img.size(),img.type(), Scalar(255)) - img;

  return findMatches( enhanced );
}
