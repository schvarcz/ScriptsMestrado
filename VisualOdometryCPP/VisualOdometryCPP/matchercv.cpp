#include "matchercv.h"

MatcherCV::MatcherCV(string methodDetector, string methodDescriptor) : IMatcher(), firstImg(true)
{
    initModule_nonfree();
    this->detector = FeatureDetector::create(methodDetector);
    this->descriptor = DescriptorExtractor::create(methodDescriptor);
}


void MatcherCV::matchFeatures()
{
//    cout << "Detecting" << endl;
    detector->detect(I1c,I1ckp);

//    cout << "Descriptors" << endl;
    descriptor->compute(I1c,I1ckp,I1cd);

    if(firstImg)
    {
        firstImg = false;
        return;
    }

//    cout << "Double matching" << endl;
    vector<DMatch> matches1,matches2, good_matches;
    flannMatcher.match(I1cd,I1pd, matches1);
    flannMatcher.match(I1pd,I1cd, matches2);

//    cout << "Matches 1: " << matches1.size() << endl;
//    cout << "Matches 2 : " << matches2.size() << endl;
//    cout << "Clear matches" << endl;
    good_matches.clear();
    float minDist=5000;
    for(vector<DMatch>::iterator it = matches1.begin();it != matches1.end(); it++)
    {
        minDist = min(minDist,it->distance);
    }
    minDist = max(8*minDist,0.002F);

    for(vector<DMatch>::iterator it = matches1.begin();it != matches1.end(); it++)
    {
        if ((it->distance <= minDist) && (matches2[it->trainIdx].trainIdx == it->queryIdx))
        {
            //add na lista!
            good_matches.push_back(*it);
        }
    }

    p_matched_2.clear();
    for(vector<DMatch>::iterator it = good_matches.begin();it != good_matches.end(); it++)
    {
        IMatcher::p_match m(I1pkp[it->trainIdx].pt.x, I1pkp[it->trainIdx].pt.y, it->trainIdx,
                            -1, -1, -1,
                            I1ckp[it->queryIdx].pt.x, I1ckp[it->queryIdx].pt.y, it->queryIdx,
                            -1, -1 ,-1);

        p_matched_2.push_back(m);
    }

//    cout << "P Matches: " << p_matched_2.size() << endl;
}

void MatcherCV::pushBack(Mat &img)
{
    if (!firstImg)
    {
        I1p = I1c;
        I1pkp.clear();
        I1pkp.swap(I1ckp);
        I1cd.copyTo(I1pd);
    }

    img.copyTo(I1c);
}
