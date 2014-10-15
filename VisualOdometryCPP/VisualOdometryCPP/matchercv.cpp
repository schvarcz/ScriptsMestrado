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

    if(I1ppkp.size() == 0)
    {
        return;
    }

//    cout << "Double matching" << endl;
    vector<DMatch>
            matches1_2,
            matches2_1,
            matches1_3,
            matches3_1,
            matches2_3,
            matches3_2;

    flannMatcher.match(I1cd,I1pd, matches1_2);
    flannMatcher.match(I1pd,I1cd, matches2_1);


    cout << "Chave: " << I1cd.size() << "\t" << I1pd.size() << "\t" << matches1_2.size() << endl;

    flannMatcher.match(I1pd,I1ppd, matches2_3);
    flannMatcher.match(I1ppd,I1pd, matches3_2);

    flannMatcher.match(I1cd,I1ppd, matches1_3);
    flannMatcher.match(I1ppd,I1cd, matches3_1);

//    vector<vector<DMatch> > matches1,matches2;
//    flannMatcher.knnMatch(I1cd, I1pd, matches1, 5);
//    flannMatcher.knnMatch(I1pd, I1cd, matches2, 5);
//    flannMatcher.radiusMatch(I1cd, I1pd, matches1,300);
//    flannMatcher.radiusMatch(I1pd,I1cd, matches2,300);

//    cout << "Matches 1: " << matches1.size() << endl;
//    cout << "Matches 2 : " << matches2.size() << endl;
//    cout << "Clear matches" << endl;


    vector<DMatch> good_matches1_2 = this->twoWayMatch(matches1_2, matches2_1);
    vector<DMatch> good_matches2_3 = this->twoWayMatch(matches2_3, matches3_2);
    vector<DMatch> good_matches1_3 = this->twoWayMatch(matches1_3, matches3_1);
    vector<DMatch> good_matches = this->threeWayMatch(good_matches1_2, good_matches2_3, good_matches1_3);
//    vector<DMatch> good_matches = this->twoWayMatchCloser(matches1, matches2);

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
        if (!this->checkAcceptance(*candidateTest, matches2[candidateTest->trainIdx], maxDist))
        {
            candidateTest->trainIdx = -1;
            //add na lista!
//            ret.push_back(*candidateTest);
        }
        ret.push_back(*candidateTest);
    }
    return ret;
}

vector<DMatch> MatcherCV::twoWayMatchCloser(vector<vector<DMatch> > matches1, vector<vector<DMatch> > matches2)
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

    DMatch candidate;
    for(vector<vector<DMatch> >::iterator it = matches1.begin();it != matches1.end(); it++)
    {
        bool found = false;
        for(vector<DMatch>::iterator candidateTest = it->begin();candidateTest != it->end(); candidateTest++)
        {
            for(vector<DMatch>::iterator correspondentTest = matches2[candidateTest->trainIdx].begin();correspondentTest != matches2[candidateTest->trainIdx].end(); correspondentTest++)
            {
                if (
                        this->checkAcceptance(*candidateTest, *correspondentTest, maxDist) &&
                        (
                            !found ||
                            sqrt(
                                pow(I1ckp[candidateTest->queryIdx].pt.x - I1pkp[candidateTest->trainIdx].pt.x,2)
                                + pow(I1ckp[candidateTest->queryIdx].pt.y - I1pkp[candidateTest->trainIdx].pt.y,2))
                            <
                            sqrt(
                                pow(I1ckp[candidate.queryIdx].pt.x - I1pkp[candidate.trainIdx].pt.x,2)
                                + pow(I1ckp[candidate.queryIdx].pt.y - I1pkp[candidate.trainIdx].pt.y,2))
                            ) &&
                        true )
                {
                    found = true;
                    candidate = *candidateTest;
                }
            }
        }
        //add na lista!
        if (found)
            ret.push_back(candidate);
    }
    return ret;
}

vector<DMatch> MatcherCV::twoWayMatchCloser(vector<DMatch>matches1, vector<DMatch> matches2)
{
    //Este método está aqui apenas para manter o padrão de nomes
    return this->twoWayMatch(matches1, matches2);
}

vector<DMatch> MatcherCV::threeWayMatch(vector<DMatch>matches1_2, vector<DMatch> matches2_3, vector<DMatch> matches1_3)
{
    vector<DMatch> ret;

    for(vector<DMatch>::iterator candidateTest = matches1_2.begin();candidateTest != matches1_2.end(); candidateTest++)
    {
        if(
                candidateTest->trainIdx != -1 &&
                matches2_3[candidateTest->trainIdx].trainIdx != -1 &&
                matches2_3[candidateTest->trainIdx].trainIdx == matches1_3[candidateTest->queryIdx].trainIdx
                )
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
//            candidateTest.distance <= 40 &&
//            sqrt(
//                pow(I1ckp[candidateTest.queryIdx].pt.x - I1pkp[candidateTest.trainIdx].pt.x,2)
//                + pow(I1ckp[candidateTest.queryIdx].pt.y - I1pkp[candidateTest.trainIdx].pt.y,2)) <= 500 &&
            true;
}

void MatcherCV::pushBack(Mat &img, bool replace)
{
    if (!replace)
    {
        if(I1pkp.size() != 0)
        {
            I1pp = I1p;
            I1ppkp.clear();
            I1ppkp.swap(I1pkp);
            I1pd.copyTo(I1ppd);
        }

        I1p = I1c;
        I1pkp.clear();
        I1pkp.swap(I1ckp);
        I1cd.copyTo(I1pd);
    }

    img.copyTo(I1c);
}
