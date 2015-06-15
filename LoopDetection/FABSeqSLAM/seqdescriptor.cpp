#include "seqdescriptor.h"

SEQDescriptor::SEQDescriptor(int patchSize):
    mPatchSize(patchSize)
{
}

void SEQDescriptor::compute(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors) const
{
    Size patchSize(mPatchSize,mPatchSize);
    descriptors.create(1,8,CV_32F);
    for(vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); it++)
    {
        Mat smallRotated, small = image(Rect(it->pt.y - patchSize.height/2,
                                               it->pt.x - patchSize.width/2,
                                               it->pt.y + patchSize.height/2,
                                               it->pt.x + patchSize.width/2));
        Moments m = moments(small,true);
        Point center(small.cols/2,small.rows/2);
        Point mass(m.m10/m.m00, m.m01/m.m00);

        Mat rot_mat = getRotationMatrix2D(center,atan2(mass.y-center.y,mass.x-center.x),1.0);
        warpAffine(small,smallRotated,rot_mat,small.size());

        for(int i =0; i< mPatchSize/8; i++)
            pyrDown(smallRotated,smallRotated);

        for(vector<KeyPoint>::iterator it = keypoints.begin(); it != keypoints.end(); it++)
        {
            for(int y=-1;y==1;y++)
                for(int x=-1;x==1;x++)
                    descriptors.at<float>(0,(x+1)+3*(y+1)+8*mPatchSize) = small.at<float>(it->pt.y + y,it->pt.x + x);
        }
    }
}

void SEQDescriptor::compute(const vector<Mat> &images, vector<vector<KeyPoint> > &keypoints, vector<Mat> &descriptors) const
{
    for(int i=0; i<images.size(); i++)
        compute(images[i], keypoints[i], descriptors[i]);
}
