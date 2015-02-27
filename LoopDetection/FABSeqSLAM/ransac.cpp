#include "ransac.h"

Ransac::Ransac(LSE model, int threshold, int iterations, int minInliers, int sampleSize):
    model(model),
    threshold(threshold),
    iterations(iterations),
    minInliers(minInliers),
    sampleSize(sampleSize)
{
}

Vector< int > Ransac::randomSamples(Mat pts)
{
    Vector <int> random_selected;

    while(true)
    {
        int random = rand()%pts.rows;
        bool ok = true;
        for (int j=0;j< random_selected.size(); j++)
        {
            if (random_selected[j] == random)
                ok = false;
        }
        if (ok)
            random_selected.push_back(random);
        if (random_selected.size() == sampleSize)
            break;
    }

    return random_selected;
}

Mat Ransac::randomPts(Mat pts)
{
    Vector <int> random_selected = randomSamples(pts);
    Mat ret;

    for(int i=0; i<random_selected.size(); i++)
        ret.push_back(pts.row(random_selected[i]));

    return ret;
}

Mat Ransac::compute(Mat pts)
{
    Mat bestInliers(0,0,CV_32F), bestParams;
    for (int i=0; i<iterations; i++)
    {
        Mat ptsSelecteds = randomPts(pts);
        Mat params = model.model(ptsSelecteds);
        Mat inliers = model.fit(pts,params,threshold);
        if (inliers.rows > bestInliers.rows)
        {
            bestInliers = inliers;
            bestParams = params;
        }

        if (bestInliers.rows > minInliers)
            break;
    }

    Mat params = model.model(bestInliers);
    return model.fit(pts,params,threshold);
}
