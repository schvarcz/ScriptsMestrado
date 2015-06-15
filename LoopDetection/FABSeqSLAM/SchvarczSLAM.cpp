#include "SchvarczSLAM.h"

SchvaczSLAM::SchvaczSLAM()
{
    init();
    BOWType = BOW_TFIDF_FREQ;
}

SchvaczSLAM::SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor):
    detector(detector),
    extractor(extractor)
{
    init();
    BOWType = BOW_TFIDF_FREQ;
}

SchvaczSLAM::SchvaczSLAM(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Mat vocab, Mat BOWIDFWeights):
    detector(detector),
    extractor(extractor),
    vocab(vocab),
    BOWIDFWeights(BOWIDFWeights)
{
    init();
}

void SchvaczSLAM::init()
{
    minVelocity         = 0.8;
    maxVelocity         = 1.2;
    RWindow             = 20;
    maxVar              = 0.7;
    maxHalfWindowMeanShiftSize = 10;
}

Mat SchvaczSLAM::apply(vector<Mat> QueryImages, vector<Mat> TestImages)
{
    occurrence = calcDifferenceMatrix(QueryImages, TestImages);
    return findMatches4(occurrence);
}

Mat SchvaczSLAM::apply(VideoCapture QueryImages, VideoCapture TestImages)
{
    occurrence = calcDifferenceMatrix(QueryImages, TestImages);
    return findMatches4(occurrence);
}

Mat SchvaczSLAM::calcDifferenceMatrix(vector<Mat> &QueryImages, vector<Mat> &TestImages)
{
    if (vocab.rows)
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

    Mat occurrence(QueryImages.size(), TestImages.size(),CV_32F);
    for(int query=0; query<occurrence.rows; query++)
    {
//        for(int test=0; test<occurrence.cols; test++)
//            occurrence.at<float>(query,test) = calcDistance(QueryImages.at(query), TestImages.at(test));

    }

    return occurrence;
}

Mat SchvaczSLAM::calcDifferenceMatrix(VideoCapture &QueryImages, VideoCapture &TestImages)
{
    if (vocab.rows)
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

    vector<Mat> queryDescs = getFeaturesDescs(QueryImages);
    vector<Mat> testDescs = getFeaturesDescs(TestImages);

    vector<BFMatcher> testDescsMatchers;

    for(int test=0; test<testDescs.size(); test++)
    {
        BFMatcher matcher(NORM_HAMMING,true);
        cout << "\r " << 100.0*((float)test / testDescs.size()) << "%         " << test << " - " << testDescs.at(test).cols;
        fflush(stdout);
        if (testDescs.at(test).cols != 0)
            matcher.add(testDescs.at(test));
        testDescsMatchers.push_back(matcher);
    }

    Mat occurrence(queryDescs.size(), testDescs.size(),CV_32F);

    cout << "Calculating the similarity matrix... " << endl;
    for(int query=0; query<occurrence.rows; query++)
    {
        float queryPerct = 100.0*((float)query / occurrence.rows);
        float maxDist = 0;
        for(int test=0; test<occurrence.cols; test++)
        {

            cout << "\r " << queryPerct + 100.0*((float)test / (occurrence.cols*occurrence.rows)) << "%         " << testDescs.at(test).cols;
            fflush(stdout);

            float dist;

            if (testDescs.at(test).cols != 0)
                dist = calcDistance(queryDescs.at(query), testDescsMatchers.at(test));
            else
                dist = numeric_limits<float>::max();
            //float dist = calcDistance(queryDescs.at(query), testDescs.at(test));
            occurrence.at<float>(query,test) = dist;
            if (dist != numeric_limits<float>::max())
                maxDist = std::max(maxDist,dist);
        }
        for(int test=0; test<occurrence.cols; test++)
            if (occurrence.at<float>(query,test) == numeric_limits<float>::max())
                occurrence.at<float>(query,test) = maxDist;
    }

    return occurrence;
}


float SchvaczSLAM::calcDistance(Mat queryDescs, BFMatcher matcher)
{
    if (queryDescs.cols == 0)
    {
        return numeric_limits<float>::max();
    }

    vector<DMatch> matches;
    matcher.match(queryDescs,matches);
    float totalDistance = 0.0f;
    for(int i=0;i<matches.size(); i++)
    {
        DMatch match = matches[i];
        totalDistance += match.distance;
    }
    return totalDistance;
}

vector<Mat> SchvaczSLAM::getFeaturesDescs(VideoCapture &movie)
{
    cout << "Extracting Descriptors... " << endl;
    Mat frame;
    vector<Mat> movieDescs;
    while(movie.read(frame))
    {
        cout << "\r " << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                                     movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%         ";
        fflush(stdout);


        Mat descs;
        vector<KeyPoint> kpts;
        detector->detect(frame,kpts);
        extractor->compute(frame,kpts,descs);
        movieDescs.push_back(descs);
    }
    cout << endl;
    return movieDescs;
}

Mat SchvaczSLAM::generateBOWImageDescs(VideoCapture movie, int BOW_TYPE)
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


    //cout << "SchvarczSLAM: Extracting Bag-of-words Image Descriptors" << endl;

    Mat frame;
    while (movie.read(frame)) {

        //cout << "\r " << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
        //                        movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%         ";

        generateBOWImageDescs(frame, schvarczSLAMTrainData, bide, BOW_TYPE);

        //fflush(stdout);
    }
    //cout << "Done" << endl;

    return schvarczSLAMTrainData;
}

Mat SchvaczSLAM::generateBOWImageDescs(vector<Mat> dataset, int BOW_TYPE)
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

Mat SchvaczSLAM::generateBOW(Mat frame, BOWImgDescriptorExtractor bide)
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

Mat SchvaczSLAM::generateBOWImageDescs(Mat frame, Mat &schvarczSLAMTrainData, BOWImgDescriptorExtractor bide, int BOW_TYPE)
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

    //cout <<  kpts.size() << " keypoints detected... " << "     ";


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

Mat SchvaczSLAM::findMatch3( Mat& re )
{
    float retmin = numeric_limits<float>::max();
    float retdeg = 0;
    for(float deg = 0.0; deg<90.0; deg+=0.5)
    {
        float soma = 0.0;
        for(int y=0;y<=re.rows; y++)
        {
            int x = y*cos(deg*M_PI/180);
            soma += re.at<float>(Point(x,y));
        }
        if (soma < retmin)
        {
            retmin = soma;
            retdeg = deg;
        }
    }
    retmin /= re.rows;
    float rety = re.rows/2;
    float retx = rety*cos(retdeg*M_PI/180);

    Mat ret(1,3,CV_32FC1,Scalar(255));
    ret.at<float>(0,0) = retmin;
    ret.at<float>(0,1) = retx;
    ret.at<float>(0,2) = rety;
    return ret;
}

Mat SchvaczSLAM::findMatches3( Mat& diff_mat )
{
    Mat diff_mat3C;
    cvtColor(diff_mat,diff_mat3C,CV_GRAY2BGR);
    Mat matches(diff_mat.rows,2,CV_32F,Scalar(numeric_limits<float>::max()));
    for(int y=0; y<=diff_mat.rows-RWindow; y++)
    {
        float matchSum = numeric_limits<float>::max();
        Mat match(1,2,CV_32F,Scalar(numeric_limits<float>::max()));
        for(int x=0; x<=diff_mat.cols-RWindow; x++)
        {
            Rect roi(x, y, RWindow, RWindow);
            Mat re;
            diff_mat(roi).convertTo(re,CV_32F);

            double mi, ma;
            minMaxLoc(re,&mi,&ma);
            re = (re-mi)/(ma-mi);

            Scalar reMean = mean(re);

//            Mat analyze;
//            diff_mat3C.copyTo(analyze);
//            rectangle(analyze,roi,Scalar(255,0,0));
//            imshow("Analisando", re);
//            imshow("Matches",analyze);
//            waitKey(33);

            if (reMean.val[0] > maxVar)
            {
                Mat matchCandidate = findMatch3(re);
                if (matchCandidate.at<float>(0) < matchSum)
                {
                    matchSum = matchCandidate.at<float>(0);
                    match.at<float>(0) = (int)(matchCandidate.at<float>(1)+x);
                    match.at<float>(1) = matchSum; //matchCandidate.at<float>(2)+y;
                }
            }
        }
        match.copyTo(matches.row(y+RWindow/2));
        //cout << retmatch.at<float>(0) << " - " << retmatch.at<float>(1) << endl;
        //circle(diff_mat3C,Point(match.at<float>(0),match.at<float>(1)),1,Scalar(255,0,0),-1);
    }
    return matches.t();
}

void SchvaczSLAM::moveLine(Vector< Point > &line,int idx, int desv, int max)
{
    for(int i=idx; i < line.size();i++)
    {
        line[i].x = MIN(round(line[i].x+desv), max);
    }
}

Mat SchvaczSLAM::vectorToMat(Vector < Point > pts)
{
    Mat ret(pts.size(), 2, CV_32F, 0.0);

    for (int i=0; i < pts.size(); i++)
    {
        ret.at<float>(i,0) = pts[i].x;
        ret.at<float>(i,1) = pts[i].y;
    }

    return ret;
}

Vector < Point > SchvaczSLAM::matToVector(Mat pts)
{
    Vector < Point > ret;

    for (int i=0; i <pts.rows; i++)
        ret.push_back(Point(pts.at<float>(i,0), pts.at<float>(i,1)));

    return ret;
}

float SchvaczSLAM::lineRank(Mat img, Mat pts)
{
    float mean = 0;

    for(int i=0; i<pts.rows; i++)
        mean += img.at<float>(pts.at<float>(i,1),pts.at<float>(i,0));

    mean /= pts.rows;

    return mean;
}

void SchvaczSLAM::findMatch4( Mat& roi , Vector< Point > &line, bool d)
{
    int xMax = roi.cols - 1;
    int yMax = roi.rows - 1;
    for(int idx =0; idx< line.size(); idx++)
    {
        Point pt = line[idx];
        Point oldPt = pt;
        float oldDesv = 0;
        while(true)
        {
            float windowSizeX = MIN(maxHalfWindowMeanShiftSize - MAX(maxHalfWindowMeanShiftSize - pt.x,0),
                                    maxHalfWindowMeanShiftSize - MAX(maxHalfWindowMeanShiftSize + pt.x - xMax,0));
            float windowSizeY = MIN(maxHalfWindowMeanShiftSize - MAX(maxHalfWindowMeanShiftSize - pt.y,0),
                                    maxHalfWindowMeanShiftSize - MAX(maxHalfWindowMeanShiftSize + pt.y - yMax,0));

            Rect window(pt.x-windowSizeX,pt.y-windowSizeY,
                        2*windowSizeX+1, 2*windowSizeY+1);

            Mat meanShiftWindow = roi(window);

            Mat histo(meanShiftWindow.cols, 1, CV_32F, Scalar(0));
            Mat range(1, meanShiftWindow.cols, CV_32F, Scalar(0));
            for(int xx = 0; xx<meanShiftWindow.cols; xx++)
            {
                histo.at<float>(xx) = sum(meanShiftWindow.col(xx))(0);
                range.at<float>(0,xx) = xx + window.x;
            }

            Mat aux = (range*histo);
            float newMeanshiftCenter = (aux.at<float>(0)/sum(histo)(0));

            float desv = newMeanshiftCenter - pt.x;

            float newX = round(desv+pt.x);
            if (newX == oldPt.x)
            {
                if (fabs(oldDesv) > fabs(desv))
                {
                    pt.x = newX;
                    moveLine(line, idx, desv, xMax);
                }
                break;
            }
            else if (newX == pt.x)
                break;

            oldPt = pt;
            pt.x = newX;
            oldDesv = desv;
            moveLine(line, idx, desv, xMax);
//            if (d)
//                draw(roi,line);
            //line[idx] = pt;
        }
    }
}


void SchvaczSLAM::draw(Mat img, Vector< Point > pts)
{
    Mat drawImg; img.copyTo(drawImg);
    cvtColor(drawImg, drawImg, CV_GRAY2RGB);
    for (int i=0; i<pts.size(); i++)
        circle(drawImg,pts[i],1,Scalar(255,0,0),-1);
    imshow("Line", drawImg);
    //waitKey();
}

Mat SchvaczSLAM::findMatches4( Mat& diff_mat )
{
    Mat diff_mat3C;
    cvtColor(diff_mat,diff_mat3C,CV_GRAY2BGR);

    //Mat imgMedias(diff_mat.rows,diff_mat.cols,CV_32F,Scalar(0.0));
    Mat matches(diff_mat.rows,2,CV_32F,Scalar(numeric_limits<float>::max()));
    for(int y=0; y < diff_mat.rows-RWindow - 2*maxHalfWindowMeanShiftSize; y++)
    {
        float matchSum = numeric_limits<float>::max();
        float bestRawMatch = numeric_limits<float>::max(), bestLSEMatch = numeric_limits<float>::max();
        Mat match(1,2,CV_32F,Scalar(numeric_limits<float>::max()));
        for(int x=0; x < diff_mat.cols-RWindow - 2*maxHalfWindowMeanShiftSize; x++)
        {
            Rect roi(x, y,
                     RWindow + 2*maxHalfWindowMeanShiftSize, RWindow + 2*maxHalfWindowMeanShiftSize);
            Mat imgRect;

            diff_mat(roi).convertTo(imgRect,CV_32F);

            double mi, ma;
            minMaxLoc(imgRect,&mi,&ma);
            imgRect = (imgRect-mi)/(ma-mi);

            Scalar reMean = mean(imgRect);

            imgRect = -imgRect + 1;

            if (reMean.val[0] > maxVar)
            {
                //imgMedias.at<float>(y,x) = 255*reMean.val[0];
                Vector < Point > line ;
                for(int i=0;i<RWindow;i++)
                {
                    line.push_back(Point(i+maxHalfWindowMeanShiftSize,
                                         i+maxHalfWindowMeanShiftSize));
                }

                //draw(imgRect,line);
                findMatch4(imgRect, line);

                Mat pts = vectorToMat(line);
                float rawMatch = 1 - lineRank(imgRect,pts);

                LSE lse;
                Mat params = lse.model(pts);
                float lseMatch = 1 - lineRank(imgRect,lse.compute(pts,params));

                Ransac ransac(lse);
                pts = ransac.compute(pts);
                line = matToVector(pts);
                float currentMatchSum = 1 - lineRank(imgRect,pts);

                if (currentMatchSum < matchSum)
                {
                    //draw(imgRect,line);
                    matchSum = currentMatchSum;
                    bestRawMatch = rawMatch;
                    bestLSEMatch = lseMatch;
                    //Scalar me = mean(pts.col(0));
                    //match.at<float>(0) = (int)(me.val[0]-maxHalfWindowMeanShiftSize+x);
                    match.at<float>(0) = (int)(pts.at<float>(pts.rows/2,0)+x);
                    match.at<float>(1) = matchSum;
                }
            }
        }
        //imwrite("media.png",imgMedias);
        //cout << "LineRank: " << bestRawMatch << " - " << bestLSEMatch << " - " << matchSum << endl;
        match.copyTo(matches.row(y+RWindow/2+maxHalfWindowMeanShiftSize));
        //cout << retmatch.at<float>(0) << " - " << retmatch.at<float>(1) << endl;
        //circle(diff_mat3C,Point(match.at<float>(0),match.at<float>(1)),1,Scalar(255,0,0),-1);
    }
    return matches.t();
}
