#include "SchvarczSLAM.h"

SchvaczSLAM::SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor):
    detector(detector),
    extractor(extractor)
{
    init();
}

SchvaczSLAM::SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab):
    detector(detector),
    extractor(extractor),
    vocab(vocab)
{
    init();
}

void SchvaczSLAM::init()
{
    minVelocity         = 0.8;
    maxVelocity         = 1.2;
    RWindow             = 10;
}

Mat SchvaczSLAM::apply(vector<Mat> QueryImages, vector<Mat> TestImages)
{
    occurrence = calcDifferenceMatrix(QueryImages, TestImages);
    return findMatches2(occurrence);
}

Mat SchvaczSLAM::calcDifferenceMatrix(vector<Mat> &QueryImages, vector<Mat> &TestImages)
{
    Mat BOWQuery = generateBOWImageDescs(QueryImages);
    Mat BOWTest = generateBOWImageDescs(TestImages);


    Mat occurrence  = BOWQuery * BOWTest.t();

    double mi, ma;
    minMaxLoc(occurrence,&mi,&ma);
    occurrence = (occurrence-mi)/ma;

    minMaxLoc(occurrence,&mi,&ma);
    occurrence = -(occurrence - ma);
    return occurrence;
}

Mat SchvaczSLAM::generateBOWImageDescs(vector<Mat> dataset)
{
    //use a FLANN matcher to generate bag-of-words representations
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);


    //extract image descriptors
    Mat schvarczSLAMTrainData;
    cout << "SchvarczSLAM: Extracting Bag-of-words Image Descriptors" << endl;

    Mat bow;
    vector<KeyPoint> kpts;

    for(vector<Mat>::iterator it = dataset.begin(); it != dataset.end(); it++)
    {
        detector->detect(*it, kpts);
        bide.compute(*it, kpts, bow);

        schvarczSLAMTrainData.push_back(bow);

        cout << "\r " << kpts.size() << " keypoints detected... " << 100.0 * ((float)(it- dataset.begin()) /
                              (float)(dataset.end()-dataset.begin())) << "%             ";
        fflush(stdout);

    }
    cout << "Done                                       " << endl;

    return schvarczSLAMTrainData;
}


/**
 * Given the difference matrix, and N index, find the image that
 * has a good match within the matching distance from image N
 * This method returns the matching index, and its score
 */
pair<int, double> SchvaczSLAM::findMatch( Mat& diff_mat, int N, int matching_dist ) {
    int move_min = static_cast<int>( minVelocity * matching_dist);
    int move_max = static_cast<int>( maxVelocity * matching_dist);

    /* Matching is based on max and min velocity */
    Mat velocity( 1, move_max - move_min + 1, CV_64FC1 );
    double * velocity_ptr = velocity.ptr<double>(0);
    for( int i = 0; i < velocity.cols; i++ )
    {
        velocity_ptr[i] = (move_min + i * 1.0) / matching_dist;
    }
    velocity = velocity.t();


    /* Create incremental indices based on the previously calculated velocity */
    Mat increment_indices( velocity.rows, matching_dist + 1, CV_32SC1 );
    for( int y = 0; y < increment_indices.rows; y++ ) {
        int * ptr    = increment_indices.ptr<int>(y);
        double v_val = velocity.at<double>(y, 0);

        for( int x = 0; x < increment_indices.cols; x++ )
            ptr[x] = static_cast<int>(floor(x * v_val));
    }


    int y_max = diff_mat.rows;

    /* Start trajectory */
    int n_start = N - (matching_dist / 2);
    Mat x( velocity.rows, matching_dist + 1, CV_32SC1 );
    for( int i = 0; i < x.cols; i++ )
        x.col(i) = (n_start + i - 1) * y_max;


    vector<float> score(diff_mat.rows - matching_dist);

    /* Perform the trajectory search to collect the scores */
    for( int s = 0; s < diff_mat.rows - matching_dist; s++ ) {
        Mat y = increment_indices + s;
        Mat( y.size(), y.type(), Scalar(y_max) ).copyTo( y, y > y_max );
        Mat idx_mat = x + y;

        float min_sum = std::numeric_limits<float>::max();
        for( int row = 0; row < idx_mat.rows; row++ ) {
            float sum = 0.0;

            for( int col = 0; col < idx_mat.cols; col++ ){
                int idx = idx_mat.at<int>(row, col);
                sum += diff_mat.at<float>( idx % y_max, idx / y_max );
            }
            min_sum = MIN( min_sum, sum );
        }

        score[s] = min_sum;
    }

    /* Find the lowest score */
    int min_index = static_cast<int>( std::min_element( score.begin(), score.end() ) - score.begin() );
    double min_val = score[min_index];

    /* ... now discard the RWindow region from where we found the lowest score ... */
    for( int i = MAX(0, min_index - RWindow / 2); i < MIN( score.size(), min_index + RWindow / 2); i++ )
        score[i] = std::numeric_limits<double>::max();

    /* ... in order to find the second lowest score */
    double min_index_2 = static_cast<int>( std::min_element( score.begin(), score.end() ) - score.begin() );
    double min_val_2 = score[min_index_2];

    return pair<int, double> ( min_index + matching_dist / 2, min_val / min_val_2 );
}

/**
 * Return a matching matrix that consists of two rows.
 * First row is the matched image index, second row is the score (the lower the score the better it is)
 */
Mat SchvaczSLAM::findMatches( Mat& diff_mat, int matching_dist ) {
    int m_dist      = matching_dist + (matching_dist % 2); /* Make sure that distance is even */
    int half_m_dist = m_dist / 2;

    /* Match matrix consists of 2 rows, first row is the index of matched image,
     second is the score. Since the higher score the larger the difference (the weaker the match)
     we initialize them with maximum value */
    Mat matches( 2, diff_mat.cols, CV_32FC1, Scalar( std::numeric_limits<float>::max() ) );

    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);
    for( int N = half_m_dist + 1; N < (diff_mat.cols - half_m_dist); N++ ) {
        pair<int, double> match = findMatch( diff_mat, N, m_dist );

        index_ptr[N] = match.first;
        score_ptr[N] = match.second;
    }

    return matches;
}

Mat SchvaczSLAM::findMatch2( Mat& re )
{
    float retmin = numeric_limits<float>::max();
    float retdeg = 0;
    for(float deg = 0.0; deg<90.0; deg+=0.5)
    {
        float soma = 0;
        for(int y=0;y<=re.rows; y++)
        {
            int x = y*cos(deg*M_PI/180);
            soma += (int)re.at<uchar>(Point(x,y));
        }
        if (soma < retmin)
        {
            retmin = soma;
            retdeg = deg;
        }
    }

    float rety = re.rows/2;
    float retx = rety*cos(retdeg*M_PI/180);

    Mat ret(1,3,CV_32FC1,Scalar(255));
    ret.at<float>(0,0) = retmin;
    ret.at<float>(0,1) = retx;
    ret.at<float>(0,2) = rety;
    return ret;
}

Mat SchvaczSLAM::findMatches2( Mat& diff_mat )
{
    Mat diff_mat3C;
    cvtColor(diff_mat,diff_mat3C,CV_GRAY2BGR);
    Mat matches;//(5,2,CV_32F,Scalar(0));
    for (int i = 0;i<RWindow/2;i++)
    {
        Mat match(1,2,CV_32F,Scalar(numeric_limits<float>::max()));
        matches.push_back(match);
    }
    for(int y=0; y<=diff_mat.rows-RWindow; y++)
    {
        float matchSum = numeric_limits<float>::max();
        Mat match(1,2,CV_32F);
        for(int x=0; x<=diff_mat.cols-RWindow; x++)
        {
            Rect roi(x, y, RWindow, RWindow);
            Mat re = diff_mat(roi);
            //Mat analyze;
            //diff_mat3C.copyTo(analyze);
            //rectangle(analyze,roi,Scalar(255,0,0));

            //imshow("Analisando", re);
            //imshow("Matches",analyze);
            //waitKey(33);
            Mat matchCandidate = findMatch2(re);
            if (matchCandidate.at<float>(0) < matchSum)
            {
                matchSum = matchCandidate.at<float>(0);
                match.at<float>(0) = (int)(matchCandidate.at<float>(1)+x);
                match.at<float>(1) = matchSum; //matchCandidate.at<float>(2)+y;
            }
        }

        matches.push_back(match);
        //cout << retmatch.at<float>(0) << " - " << retmatch.at<float>(1) << endl;
        //circle(diff_mat3C,Point(match.at<float>(0),match.at<float>(1)),1,Scalar(255,0,0),-1);
    }
    for (int i = 0;i<RWindow/2;i++)
    {
        Mat match(1,2,CV_32F,Scalar(numeric_limits<float>::max()));
        matches.push_back(match);
    }
    //imshow("Matches2",diff_mat3C);
    //waitKey();
    return matches.t();
}
