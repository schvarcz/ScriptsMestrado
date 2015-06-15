#include "seqfeature.h"

SEQFeature::SEQFeature(int patchSize):
    Feature2D(), patchSize(patchSize)
{
    detector = new FastFeatureDetector();
}


void SEQFeature::operator ()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints, OutputArray descriptors, bool useProvidedKeypoints) const
{
    Mat img = image.getMat();
    if(!useProvidedKeypoints)
        this->detect(img,keypoints,mask.getMat());

    descriptors.create(1,this->descriptorSize(),this->descriptorType());
    Mat desc = descriptors.getMat();
    this->compute(img,keypoints,desc);
}


void SEQFeature::computeImpl(const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors) const
{
    for(vector<KeyPoint>::iterator it = keypoints.begin();it != keypoints.end();)
    {
        KeyPoint kpt = (*it);

        if (
                kpt.pt.x >= patchSize/2 &&
                kpt.pt.y >= patchSize/2 &&
                kpt.pt.x <= image.cols - patchSize/2 &&
                kpt.pt.y <= image.rows - patchSize/2)
        {
            Mat roi;
            image(Rect(kpt.pt.x - patchSize/2, kpt.pt.y- patchSize/2, patchSize, patchSize))
                    .copyTo(roi);
            roi.convertTo(roi,this->descriptorType());
            Mat std,mean;
            meanStdDev(roi,mean,std);
            roi = (roi-mean.at<float>(0))/std.at<float>(0);
            Mat desc(1,this->descriptorSize(),this->descriptorType(),Scalar(0));
            for(int i=0; i<roi.rows;i++)
                roi.row(i).copyTo(desc.colRange(i*patchSize,(1+i)*patchSize));
            descriptors.push_back(desc);
            it++;
        }
        else
        {
            keypoints.erase(it);
        }
    }

}


void SEQFeature::detectImpl(const Mat &image, vector<KeyPoint> &keypoints, const Mat &mask) const
{
    this->detector->detect(image,keypoints,mask);
}


int SEQFeature::descriptorSize() const
{
    return patchSize*patchSize;
}


int SEQFeature::descriptorType() const
{
    return CV_32F;
}

