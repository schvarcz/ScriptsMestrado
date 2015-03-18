#include "outdoorslam.h"

OutdoorSLAM::OutdoorSLAM()
{
    init();
    BOWType = BOW_TFIDF_FREQ;
}

OutdoorSLAM::OutdoorSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor):
    detector(detector),
    extractor(extractor)
{
    init();
    BOWType = BOW_TFIDF_FREQ;
}

OutdoorSLAM::OutdoorSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab, Mat BOWIDFWeights):
    detector(detector),
    extractor(extractor),
    vocab(vocab),
    BOWIDFWeights(BOWIDFWeights)
{
    init();
}

void OutdoorSLAM::init()
{
    minVelocity         = 0.8;
    maxVelocity         = 1.2;
    RWindow             = 20;
    maxVar              = 0.7;
    maxHalfWindowMeanShiftSize = 10;
}

Mat OutdoorSLAM::apply(vector<Mat> QueryImages, vector<Mat> TestImages)
{
    occurrence = calcDifferenceMatrix(QueryImages, TestImages);
    return findMatches(occurrence);
}

Mat OutdoorSLAM::apply(VideoCapture QueryImages, VideoCapture TestImages)
{
    occurrence = calcDifferenceMatrix(QueryImages, TestImages);
    //calcSequencesMatrix();
    //return Mat();
    return findMatches(occurrence);
}
Mat OutdoorSLAM::calcSequencesMatrix() //Paul Newman's shit.
{
//    Mat matches(diff_mat.rows,2,CV_32F,Scalar(numeric_limits<float>::max()));
    cout << "Paul Newman's shit" << endl;
    double penalidade = 0.1;
    double tolerancia = 0.3;
    double smooth = 0.3;
    Mat M = -occurrence.clone()+1;

    Mat H(M.rows, M.cols,CV_32F,0.0);
    Mat RecoverPath(M.rows, M.cols,CV_32F,0.0);
    for(int y=1; y<H.rows; y++)
    {
        for(int x=1; x<H.cols; x++)
        {
            H.at<float>(y,x) = M.at<float>(y,x);
            if ((H.at<float>(y-1,x-1) >= H.at<float>(y-1,x)) && (H.at<float>(y-1,x-1) >= H.at<float>(y,x-1)))
            {
                RecoverPath.at<float>(y,x) = 2.0;
                H.at<float>(y,x) = H.at<float>(y-1,x-1) + M.at<float>(y,x);
            }
            else if ((H.at<float>(y-1,x) >= H.at<float>(y-1,x-1)) && (H.at<float>(y-1,x) >= H.at<float>(y,x-1)))
            {
                RecoverPath.at<float>(y,x) = 1.0;
                H.at<float>(y,x) = H.at<float>(y-1,x-1) + M.at<float>(y,x) - penalidade;
            }
            else if ((H.at<float>(y,x-1) >= H.at<float>(y-1,x-1)) && (H.at<float>(y,x-1) >= H.at<float>(y-1,x)))
            {
                RecoverPath.at<float>(y,x) = 3.0;
                H.at<float>(y,x) = H.at<float>(y-1,x-1) + M.at<float>(y,x) - penalidade;
            }

            if (M.at<float>(y,x) < tolerancia)
                if (H.at<float>(y,x) > tolerancia)
                {
                    H.at<float>(y,x) = smooth*(H.at<float>(y,x)-M.at<float>(y,x));
                }
                else
                    H.at<float>(y,x) = 0.0;
        }
    }

    double mi, ma;
    Point ptMin, ptMax;
    minMaxLoc(H,&mi,&ma,&ptMin,&ptMax);

    Mat imgPath;
    cvtColor(255*(H-mi)/(ma-mi),imgPath,CV_GRAY2RGB);
    vector<Point> path;
    while((ptMax.x >=0) && (ptMax.y >=0) && (H.at<float>(ptMax.y,ptMax.x) > tolerancia))
    {
        Point pt(ptMax.x,ptMax.y);
        path.push_back(pt);
        circle(imgPath,pt,1,Scalar(255,0,0));
        switch ((int)RecoverPath.at<float>(ptMax.y,ptMax.x)) {
        case 1:
            ptMax.y -= 1;
            break;
        case 2:
            ptMax.x -= 1;
            ptMax.y -= 1;
        case 3:
            ptMax.x -= 1;
        default:
            break;
        }
    }
//    matches
    imshow("H",H);
    imshow("imgPath",imgPath);
    imwrite("h.bmp",H);
    imwrite("imgPath.bmp",imgPath);
    waitKey();
//    return matches;
}

Mat OutdoorSLAM::calcDifferenceMatrix(vector<Mat> &QueryImages, vector<Mat> &TestImages)
{
    Mat BOWQuery = generateBOWImageDescs(QueryImages,BOWType);
    Mat BOWTest = generateBOWImageDescs(TestImages,BOWType);

    Mat occurrence  = BOWQuery * BOWTest.t();

    double mi, ma;
    minMaxLoc(occurrence,&mi,&ma);
    occurrence = (occurrence-mi)/(ma-mi);

    minMaxLoc(occurrence,&mi,&ma);
    occurrence = -(occurrence - ma);
    return occurrence;
}

Mat OutdoorSLAM::calcDifferenceMatrix(VideoCapture &QueryImages, VideoCapture &TestImages)
{
    Mat BOWQuery = generateBOWImageDescs(QueryImages,BOWType);
    Mat BOWTest = generateBOWImageDescs(TestImages,BOWType);

    Mat occurrence  = BOWQuery * BOWTest.t();

    double mi, ma;
    minMaxLoc(occurrence,&mi,&ma);
    occurrence = (occurrence-mi)/(ma-mi);

    minMaxLoc(occurrence,&mi,&ma);
    occurrence = -(occurrence - ma);
    return occurrence;
}


Mat OutdoorSLAM::generateBOWImageDescs(VideoCapture movie, int BOW_TYPE)
{
    //extract image descriptors
    Mat schvarczSLAMTrainData;

    if (!movie.isOpened()) {
        cerr << "GenerateBOWImageDescs: movie not found" << endl;
        return schvarczSLAMTrainData;
    }

    //use a FLANN matcher to generate bag-of-words representations
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);


    cout << "SchvarczSLAM: Extracting Bag-of-words Image Descriptors" << endl;

    Mat frame;
    while (movie.read(frame)) {

        cout << "\r " << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                                movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%         ";

        generateBOWImageDescs(frame, schvarczSLAMTrainData, bide, BOW_TYPE);

        fflush(stdout);
    }
    cout << "Done" << endl;

    return schvarczSLAMTrainData;
}

Mat OutdoorSLAM::generateBOWImageDescs(vector<Mat> dataset, int BOW_TYPE)
{
    //use a FLANN matcher to generate bag-of-words representations
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
    BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //extract image descriptors
    Mat schvarczSLAMTrainData;
    cout << "SchvarczSLAM: Extracting Bag-of-words Image Descriptors" << endl;

    for(vector<Mat>::iterator it = dataset.begin(); it != dataset.end(); it++)
    {
        cout << "\r " << 100.0 * ((float)(it- dataset.begin()+1) /
                              (float)(dataset.end()-dataset.begin())) << "%   ";

        generateBOWImageDescs(*it, schvarczSLAMTrainData, bide, BOW_TYPE);

        fflush(stdout);
    }
    cout << "Done                                       " << endl;

    return schvarczSLAMTrainData;
}

Mat OutdoorSLAM::generateBOW(Mat frame, BOWImgDescriptorExtractor bide)
{
    Mat result = frame.clone();
    if ( result.channels() > 1 )
        cvtColor( result, result, CV_BGR2GRAY );

    Size patchSize(160,80);
    Mat patch;
    double ma, mi;

    Mat bowReturn = Mat::zeros(1, BOWIDFWeights.cols, CV_32F);


    for(int y = 0; y < result.rows; y+= patchSize.height ) {
        for(int x = 0; x < result.cols; x+= patchSize.width ) {
            /* Extract patch */
            patch = result(Rect(x, y, patchSize.width, patchSize.height));

            minMaxIdx(patch,&mi,&ma);
            patch=255*(patch-mi)/(ma-mi);

            vector<KeyPoint> kpts;
            Mat bow;

            detector->detect(patch, kpts);
            bide.compute(patch, kpts, bow);

            if (bow.cols != 0)
                bowReturn+= bow;
        }
    }

    return bowReturn;
}

Mat OutdoorSLAM::generateBOWImageDescs(Mat frame, Mat &schvarczSLAMTrainData, BOWImgDescriptorExtractor bide, int BOW_TYPE)
{
    vector< vector< int > > pointIdxsOfClusters;
    vector<KeyPoint> kpts;
    Mat bow;

    detector->detect(frame, kpts);
    bide.compute(frame, kpts, bow, &pointIdxsOfClusters);
    //bow = generateBOW(frame,bide);
    if (bow.cols == 0)
    {
        bow = Mat::zeros(1, BOWIDFWeights.cols, CV_32F);
    }

    if (BOW_TYPE == BOW_NORM)
    {
        schvarczSLAMTrainData.push_back(bow);
    }
    else if (BOW_TYPE == BOW_FREQ)
    {
        Mat bow2(1,pointIdxsOfClusters.size(),CV_32F,Scalar(0));
        for (int i =0; i < pointIdxsOfClusters.size();i++)
        {
            bow2.at<float>(0,i) = (float)pointIdxsOfClusters[i].size();
        }
        schvarczSLAMTrainData.push_back(bow2);
    }
    else if (BOW_TYPE == BOW_TFIDF_NORM)
    {
        multiply(bow,BOWIDFWeights,bow);
        schvarczSLAMTrainData.push_back(bow);
    }
    else if (BOW_TYPE == BOW_TFIDF_FREQ)
    {
        Mat bow2(1,pointIdxsOfClusters.size(),CV_32F,Scalar(0));
        for (int i =0; i < pointIdxsOfClusters.size();i++)
        {
            bow2.at<float>(0,i) = (float)pointIdxsOfClusters[i].size();
        }
        multiply(bow2,BOWIDFWeights,bow2);
        schvarczSLAMTrainData.push_back(bow2);
    }

    cout <<  kpts.size() << " keypoints detected... " << "     ";


    return schvarczSLAMTrainData;
}

/**
 * Given the difference matrix, and N index, find the image that
 * has a good match within the matching distance from image N
 * This method returns the matching index, and its score
 */
pair<int, double> OutdoorSLAM::findMatch( Mat& diff_mat, int N, int matching_dist ) {
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
                sum += diff_mat.at<float>( idx / y_max, idx % y_max );
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
Mat OutdoorSLAM::findMatches( Mat& diff_mat, int matching_dist ) {
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

void OutdoorSLAM::moveLine(Vector< Point > &line,int idx, int desv, int max)
{
    for(int i=idx; i < line.size();i++)
    {
        line[i].x = MIN(round(line[i].x+desv), max);
    }
}

Mat OutdoorSLAM::vectorToMat(Vector < Point > pts)
{
    Mat ret(pts.size(), 2, CV_32F, 0.0);

    for (int i=0; i < pts.size(); i++)
    {
        ret.at<float>(i,0) = pts[i].x;
        ret.at<float>(i,1) = pts[i].y;
    }

    return ret;
}

Vector < Point > OutdoorSLAM::matToVector(Mat pts)
{
    Vector < Point > ret;

    for (int i=0; i <pts.rows; i++)
        ret.push_back(Point(pts.at<float>(i,0), pts.at<float>(i,1)));

    return ret;
}

float OutdoorSLAM::lineRank(Mat img, Mat pts)
{
    float mean = 0;
    for(int i=0; i<pts.rows; i++)
    {
        //cout << pts.at<float>(i,1) << "x" << pts.at<float>(i,0) << " - " << pts.rows << endl;
        mean += img.at<float>(pts.at<float>(i,1),pts.at<float>(i,0));
    }
    mean /= pts.rows;

    return mean;
}

void OutdoorSLAM::draw(Mat img, Vector< Point > pts)
{
    Mat drawImg; img.copyTo(drawImg);
    cvtColor(drawImg, drawImg, CV_GRAY2RGB);
    for (int i=0; i<pts.size(); i++)
        circle(drawImg,pts[i],1,Scalar(255,0,0),-1);
    imshow("Line", drawImg);
    waitKey();
}
