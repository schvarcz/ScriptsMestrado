#include "lse.h"

LSE::LSE()
{
}

Mat LSE::model(Mat points)
{
    Mat A(points.rows,4,CV_32F,1);
    Mat B = points.col(1);

    for(int y=0; y<A.rows;y++)
    {
        for(int exp=A.cols-1;exp>0; exp--)
            A.at<float>(y,exp) = pow(points.at<float>(y,0), exp);
    }

    Mat params(A.cols,1,CV_32F, 0);

    solve(A,B,params,DECOMP_NORMAL);

    return params;
}

Mat LSE::fit(Mat points, Mat params, float threshold)
{
    Mat y = points.col(1);
    Mat x = points.col(0);

    Mat sum(x.rows,x.cols,CV_32F,0);
    for(int i =0; i <params.rows; i++)
    {
        Mat exp;
        pow(x, i, exp);
        sum = sum + exp*params.at<float>(i);
    }

    Mat erros = y - sum;
    Mat inliers;

    for(int i =0; i<erros.rows; i++)
    {
        if (fabs(erros.at<float>(i)) < threshold)
        {
            Mat row;
            points.row(i).copyTo(row);
            inliers.push_back(row);
        }
    }

    return inliers;
}


Mat LSE::compute(Mat points, Mat params)
{
    Mat newPts;
    points.copyTo(newPts);
    Mat x = points.col(0);

    Mat sum(x.rows,x.cols,CV_32F,0);
    for(int i =0; i <params.rows; i++)
    {
        Mat exp;
        pow(x, i, exp);
        sum = sum + exp*params.at<float>(i);
    }

    sum.copyTo(newPts.col(1));

    return newPts;
}
