#include "imatcher.h"

IMatcher::IMatcher()
{
}


void IMatcher::bucketFeatures(int32_t max_features,float bucket_width,float bucket_height) {

  // find max values
  float u_max = 0;
  float v_max = 0;
  for (vector<IMatcher::p_match>::iterator it = p_matched_2.begin(); it!=p_matched_2.end(); it++) {
    if (it->u1c>u_max) u_max=it->u1c;
    if (it->v1c>v_max) v_max=it->v1c;
  }

  // allocate number of buckets needed
  int32_t bucket_cols = (int32_t)floor(u_max/bucket_width)+1;
  int32_t bucket_rows = (int32_t)floor(v_max/bucket_height)+1;
  vector<IMatcher::p_match> *buckets = new vector<IMatcher::p_match>[bucket_cols*bucket_rows];

  // assign matches to their buckets
  for (vector<IMatcher::p_match>::iterator it=p_matched_2.begin(); it!=p_matched_2.end(); it++) {
    int32_t u = (int32_t)floor(it->u1c/bucket_width);
    int32_t v = (int32_t)floor(it->v1c/bucket_height);
    buckets[v*bucket_cols+u].push_back(*it);
  }

  // refill p_matched from buckets
  p_matched_2.clear();
  for (int32_t i=0; i<bucket_cols*bucket_rows; i++) {

    // shuffle bucket indices randomly
    std::random_shuffle(buckets[i].begin(),buckets[i].end());

    // add up to max_features features from this bucket to p_matched
    int32_t k=0;
    for (vector<IMatcher::p_match>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++) {
      p_matched_2.push_back(*it);
      k++;
      if (k>=max_features)
        break;
    }
  }

  // free buckets
  delete []buckets;
}
