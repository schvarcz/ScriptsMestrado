#include "matchercv.h"

MatcherCV::MatcherCV(string methodDetector, string methodDescriptor) : IMatcher()
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

    if(I1pkp.size() == 0)
    {
        return;
    }

//    cout << "Double matching" << endl;
//    vector<DMatch> matches1,matches2;
//    flannMatcher.match(I1cd,I1pd, matches1);
//    flannMatcher.match(I1pd,I1cd, matches2);

    vector<vector<DMatch> > matches1,matches2;
    flannMatcher.knnMatch(I1cd, I1pd, matches1, 5);
    flannMatcher.knnMatch(I1pd, I1cd, matches2, 5);
//    flannMatcher.radiusMatch(I1cd, I1pd, matches1,300);
//    flannMatcher.radiusMatch(I1pd,I1cd, matches2,300);

//    cout << "Matches 1: " << matches1.size() << endl;
//    cout << "Matches 2 : " << matches2.size() << endl;
//    cout << "Clear matches" << endl;


    vector<DMatch> good_matches = this->twoWayMatch(matches1, matches2);

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

vector<DMatch> MatcherCV::twoWayMatch(vector<vector<DMatch> > matches1, vector<vector<DMatch> > matches2)
{
    vector<DMatch> ret;

    float maxDist=5000;
    for(vector<vector<DMatch> >::iterator candidateTest = matches1.begin();candidateTest != matches1.end(); candidateTest++)
    {
        for(vector<DMatch>::iterator possibleMatch = candidateTest->begin();possibleMatch != candidateTest->end(); possibleMatch++)
        {
            maxDist = min(maxDist,possibleMatch->distance);
        }
    }
    maxDist = max(10*maxDist,0.002F);

    for(vector<vector<DMatch> >::iterator it = matches1.begin();it != matches1.end(); it++)
    {
        for(vector<DMatch>::iterator candidateTest = it->begin();candidateTest != it->end(); candidateTest++)
        {
            for(vector<DMatch>::iterator correspondentTest = matches2[candidateTest->trainIdx].begin();correspondentTest != matches2[candidateTest->trainIdx].end(); correspondentTest++)
            {
                if (this->checkAcceptance(*candidateTest, *correspondentTest, maxDist))
                {
                    //add na lista!
                    ret.push_back(*candidateTest);
                    correspondentTest = matches2[candidateTest->trainIdx].end()-1;
                    candidateTest = it->end()-1;
                    break;
                }
            }
        }
    }
    return ret;
}

vector<DMatch> MatcherCV::twoWayMatch(vector<DMatch>matches1, vector<DMatch> matches2)
{
    vector<DMatch> ret;

    float maxDist=5000;
    for(vector<DMatch>::iterator it = matches1.begin();it != matches1.end(); it++)
    {
        maxDist = min(maxDist,it->distance);
    }
    maxDist = max(10*maxDist,0.002F);

    for(vector<DMatch>::iterator candidateTest = matches1.begin();candidateTest != matches1.end(); candidateTest++)
    {
        if (this->checkAcceptance(*candidateTest, matches2[candidateTest->trainIdx], maxDist))
        {
            //add na lista!
            ret.push_back(*candidateTest);
        }
    }
    return ret;
}

bool MatcherCV::checkAcceptance(DMatch candidateTest, DMatch correspondentTest, float maxDist)
{
    return  correspondentTest.trainIdx == candidateTest.queryIdx &&
            candidateTest.distance <= maxDist &&
            sqrt(
                pow(I1ckp[candidateTest.queryIdx].pt.x - I1pkp[candidateTest.trainIdx].pt.x,2)
                + pow(I1ckp[candidateTest.queryIdx].pt.y - I1pkp[candidateTest.trainIdx].pt.y,2)) <= 200 &&
            true;
}

void MatcherCV::pushBack(Mat &img, bool replace)
{
    if (!replace)
    {
        I1p = I1c;
        I1pkp.clear();
        I1pkp.swap(I1ckp);
        I1cd.copyTo(I1pd);
    }

    img.copyTo(I1c);
}
