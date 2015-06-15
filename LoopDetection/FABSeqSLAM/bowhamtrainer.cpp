#include "bowhamtrainer.h"

BOWHAMTrainer::BOWHAMTrainer(double _clusterSize) :
    clusterSize(_clusterSize) {
}

BOWHAMTrainer::~BOWHAMTrainer() {
}

Mat BOWHAMTrainer::cluster() const {
    CV_Assert(!descriptors.empty());
    int descCount = 0;

    for(size_t i = 0; i < descriptors.size(); i++)
        descCount += descriptors[i].rows;

    Mat mergedDescriptors(descCount, descriptors[0].cols, descriptors[0].type());

    for(size_t i = 0, start = 0; i < descriptors.size(); i++)
    {
        Mat submut = mergedDescriptors.rowRange((int)start,
            (int)(start + descriptors[i].rows));
        descriptors[i].copyTo(submut);
        start += descriptors[i].rows;
    }

    return cluster(mergedDescriptors);
}

Mat BOWHAMTrainer::cluster(const Mat& descriptors) const {

    CV_Assert(!descriptors.empty());

    // TODO: sort the descriptors before clustering.
    std::cout.setf(std::ios_base::fixed);
    std::cout.precision(0);

    std::cout <<  "Init centers" << std::endl;
    vector<Mat> initialCentres;
    initialCentres.push_back(descriptors.row(0));
    for (int i = 1; i < descriptors.rows; i++) {

        std::cout << "\r  Descriptors: " << i <<  "/" << descriptors.rows << "(" << 100.0*(float)i/descriptors.rows << "%)  Clusters: " << initialCentres.size();
        fflush(stdout);

        double minDist = DBL_MAX;
        for (size_t j = 0; j < initialCentres.size(); j++) {
            minDist = std::min(minDist, norm(descriptors.row(i),initialCentres[j],NORM_HAMMING));
        }
        cout << endl << descriptors.cols << " - " <<  norm(descriptors.row(i),initialCentres[0],NORM_HAMMING) << endl;
        if (minDist > clusterSize)
            initialCentres.push_back(descriptors.row(i));
    }

    std::cout << std::endl <<  "Clustering" << std::endl;
    vector<list<Mat> > clusters;
    clusters.resize(initialCentres.size());
    for (int i = 0; i < descriptors.rows; i++) {
        std::cout << "\r  Descriptors: " << i <<  "/" << descriptors.rows << "(" << 100.0*(float)i/descriptors.rows << "%)";
        fflush(stdout);

        int index; double dist, minDist = DBL_MAX;
        for (size_t j = 0; j < initialCentres.size(); j++) {
            dist = norm(descriptors.row(i),initialCentres[j],NORM_HAMMING);
            if (dist < minDist) {
                minDist = dist;
                index = j;
            }
        }
        clusters[index].push_back(descriptors.row(i));
    }

    // TODO: throw away small clusters.

    std::cout <<  "Updating centers" << std::endl;
    Mat vocabulary;
    Mat centre = Mat::zeros(1,descriptors.cols,descriptors.type());
    for (size_t i = 0; i < clusters.size(); i++) {
        std::cout << "\r  Clusters: " << i <<  "/" << clusters.size() << "(" << 100.0*(float)i/clusters.size() << ")  Vocabulary: " << vocabulary.size();
        fflush(stdout);
        centre.setTo(0);
        for (list<Mat>::iterator Ci = clusters[i].begin(); Ci != clusters[i].end(); Ci++) {
            centre += *Ci;
        }
        centre /= (double)clusters[i].size();
        vocabulary.push_back(centre);
    }

    return vocabulary;
}


