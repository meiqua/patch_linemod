#include "pose_refine.h"
#include <omp.h>
#include "linemodLevelup.h"
// ---------------------helper begin--------------------------------
template <typename T> static
std::vector<size_t> argsort(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx;
}

// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const cv::Rect_<_Tp>& a, const cv::Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float computeOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

// ---------------------helper end--------------------------------


PoseRefine::PoseRefine(std::string model_path, cv::Mat depth, cv::Mat K):
    #ifdef CUDA_ON
        tris(model.tris.size()),
    #else
        tris(model.tris),
    #endif
    model(model_path)  // model inits first
{
#ifdef CUDA_ON
    thrust::copy(model.tris.begin(), model.tris.end(), tris.begin_thr());
#endif
    if(!K.empty()) set_K(K);
    if(!depth.empty()) set_depth(depth);
}

void PoseRefine::set_depth(cv::Mat depth)
{
    assert(depth.type() == CV_16U && K.type() == CV_32F);

    cv::medianBlur(depth, depth, 5);
    scene_depth = depth;
    width = depth.cols;
    height = depth.rows;
    proj_mat = cuda_renderer::compute_proj(K, width, height);

    depth_edge = get_depth_edge(scene_depth);
    scene.init(scene_depth, depth_edge, *reinterpret_cast<Mat3x3f*>(K.data), scene_buffer);
}

void PoseRefine::set_K(cv::Mat K)
{
    assert(K.type() == CV_32F);
    this->K = K;
}

void PoseRefine::set_K_width_height(cv::Mat K, int width, int height)
{
    assert(K.type() == CV_32F);
    this->K = K;
    this->width = width;
    this->height = height;
    proj_mat = cuda_renderer::compute_proj(K, width, height);
}

std::vector<cuda_icp::RegistrationResult> PoseRefine::process_batch(std::vector<cv::Mat>& init_poses,
                                                                    int down_sample)
{
    const bool debug_ = true;
    const bool record_ = false;
    if(debug_){
        if(record_){
            cv::FileStorage fs("/home/meiqua/dump_process_batch.yml", cv::FileStorage::WRITE);
            fs << "down_sample" << down_sample;
            fs << "init_poses" << "[";
            for(auto& r: init_poses){
                fs << r;
            }
            fs << "]";

        }
        else{
            cv::FileStorage fs("/home/meiqua/dump_process_batch.yml", cv::FileStorage::READ);
            fs["down_sample"] >> down_sample;
            cv::FileNode results_fn = fs["init_poses"];
            init_poses.clear();
            init_poses.resize(results_fn.size());
            cv::FileNodeIterator it = results_fn.begin(), it_end = results_fn.end();
            for (int i_r = 0; it != it_end; ++it, ++i_r){
                (*it) >> init_poses[i_r];
            }
        }
    }

    int init_size = init_poses.size();
    std::vector<cuda_icp::RegistrationResult> result_poses(init_size);
    cuda_icp::ICPConvergenceCriteria criteria;

    assert(width%down_sample==0 && height%down_sample==0);
    const int width_local = width/down_sample;
    const int height_local = height/down_sample;

    Mat3x3f K_icp((float*)K.data); // ugly but useful
    K_icp[0][0] /= down_sample; K_icp[1][1] /= down_sample;
    K_icp[0][2] /= down_sample; K_icp[1][2] /= down_sample;

    std::vector<cuda_renderer::Model::mat4x4> mat4_v(batch_size);

    // icp is m, while init poses & renderer is mm
    auto to_mm = [](Mat4x4f& trans){
        trans[0][3] *= 1000.0f;
        trans[1][3] *= 1000.0f;
        trans[2][3] *= 1000.0f;
        return trans;
    };
    auto to_m = [](Mat4x4f& trans){
        trans[0][3] /= 1000.0f;
        trans[1][3] /= 1000.0f;
        trans[2][3] /= 1000.0f;
        return trans;
    };

    auto icp_batch = [&](int i){
        // directly down sample by viewport
        auto depths = cuda_renderer::render(tris, mat4_v, width_local, height_local, proj_mat);

        // cuda per thread stream
#pragma omp parallel num_threads(mat4_v.size())
        {
            int j = omp_get_thread_num();
            Mat4x4f temp = to_m(reinterpret_cast<Mat4x4f&>(mat4_v[j]));

            auto pcd1_cuda = cuda_icp::depth2cloud(depths.data() + j*width_local*height_local,
                                                   width_local, height_local, K_icp);

            result_poses[j+i] = cuda_icp::ICP_Point2Plane(pcd1_cuda, scene, criteria);

            temp = result_poses[j+i].transformation_ * temp;
            result_poses[j + i].transformation_ = to_mm(temp);
        }
    };

    int i=0;
    for(; i<=init_size-batch_size; i+=batch_size){
        for(int j=0; j<batch_size; j++) mat4_v[j].init_from_cv(init_poses[j+i]);
        icp_batch(i);
    }

    int left = init_size - i;
    if(left > 0){
        mat4_v.resize(left);
        for(int j=0; j<left; j++) mat4_v[j].init_from_cv(init_poses[j+i]);
        icp_batch(i);
    }

    return result_poses;
}

std::vector<cuda_icp::RegistrationResult> PoseRefine::results_filter(
        std::vector<cuda_icp::RegistrationResult> &results, float fitness_thresh, float rmse_thresh)
{
    const bool debug_ = false;
    const bool record_ = false;
    if(debug_){
        if(record_){
            cv::FileStorage fs("/home/meiqua/dump.yml", cv::FileStorage::WRITE);
            fs << "results" << "[";
            for(auto& r: results){
                fs<< "{";

                fs << "fitness" << r.fitness_;
                fs << "inlier_rmse" << r.inlier_rmse_;
                fs << "transformation" << "[";
                for(int i=0; i<4; i++){
                    for(int j=0; j<4; j++){
                        fs << r.transformation_[i][j];
                    }
                }
                fs << "]";
                fs << "}";
            }
            fs << "]";
        }
        else{
            cv::FileStorage fs("/home/meiqua/dump.yml", cv::FileStorage::READ);
            cv::FileNode results_fn = fs["results"];
            results.resize(results_fn.size());
            cv::FileNodeIterator it = results_fn.begin(), it_end = results_fn.end();
            for (int i_r = 0; it != it_end; ++it, ++i_r)
            {
                results[i_r].fitness_ = (*it)["fitness"];
                results[i_r].inlier_rmse_ = (*it)["inlier_rmse"];

                cv::FileNode trans_fn = (*it)["transformation"];
                assert(trans_fn.type() == cv::FileNode::SEQ);

                cv::FileNodeIterator fni = trans_fn.begin();
                for(int i=0; i<4; i++){
                    for(int j=0; j<4; j++){
                        results[i_r].transformation_[i][j] = float(*fni);
                        fni++;
                    }
                }
            }
        }
    }

    // first pass, check rmse & fitness
    std::vector<cuda_icp::RegistrationResult> filtered;
    std::vector<cuda_renderer::Model::mat4x4> mat4_v;

    for(auto& res: results){
        if(res.fitness_>fitness_thresh && res.inlier_rmse_<rmse_thresh){
            filtered.push_back(res);
            mat4_v.push_back(reinterpret_cast<cuda_renderer::Model::mat4x4&>(res.transformation_));
        }
    }
    if(filtered.empty()) return std::vector<cuda_icp::RegistrationResult>();

    // second pass, check edge hit
    struct Result_bbox{
        cuda_icp::RegistrationResult result;
        cv::Rect bbox;
        float score;
        bool operator > (const Result_bbox& other) const {
            return this->score > other.score;
        }
    };

    std::vector<Result_bbox> filtered2;

    const int down_sample = 2;
    assert(width%down_sample==0 && height%down_sample==0);
    const int width_local = width/down_sample;
    const int height_local = height/down_sample;

    auto depths = cuda_renderer::render_host(tris, mat4_v, width_local, height_local, proj_mat);

#pragma omp declare reduction (merge : std::vector<Result_bbox> : \
    omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

#pragma omp parallel for reduction(merge: filtered2)
    for(int i=0; i<filtered.size(); i++){
        cv::Mat depth_int(height_local, width_local, CV_32SC1, depths.data() + i*width_local*height_local);
        cv::Mat mask = depth_int > 0;

        cv::Mat mask_erode;
        cv::dilate(mask, mask_erode, cv::Mat());
        cv::Mat mask_edge;
        cv::bitwise_xor(mask, mask_erode, mask_edge);

        cv::Mat Points;
        cv::findNonZero(mask_edge, Points);
        auto bbox = boundingRect(Points);

        Result_bbox temp;
        temp.result = filtered[i];
        temp.bbox = bbox;
        temp.score =  1/(10 * filtered[i].inlier_rmse_);
        filtered2.push_back(temp);
    }

    // third pass, nms
    std::vector<cuda_icp::RegistrationResult> filtered3;
    if(filtered2.empty()) return filtered3;
    if(filtered2.size() == 1){
        filtered3.push_back(filtered2[0].result);
        return filtered3;
    }

    // nms
    float nms_thrsh = 0.5f;
    auto sorted_idx = argsort(filtered2);
    std::vector<size_t> out_idx;

    // for all bbox
    for(size_t i=0; i<sorted_idx.size(); i++){
        size_t idx = sorted_idx[i];
        bool keep = true;

        // not overlap with existed bbox
        for(size_t j=0; j<out_idx.size() && keep; j++){
            size_t kept_idx = out_idx[j];
            float overlap = computeOverlap(filtered2[idx].bbox, filtered2[kept_idx].bbox);
            keep = overlap <= nms_thrsh;
        }
        if(keep){
            out_idx.push_back(idx);
        }
    }

    filtered3.resize(out_idx.size());
    for(size_t i=0; i<out_idx.size(); i++){
        filtered3[i] = filtered2[out_idx[i]].result;
    }

    return filtered3;
}

template<typename F>
auto PoseRefine::render_what(F f, std::vector<cv::Mat> &init_poses, int down_sample)
{
    assert(width%down_sample==0 && height%down_sample==0);
    const int width_local = width/down_sample;
    const int height_local = height/down_sample;

    std::vector<cuda_renderer::Model::mat4x4> mat4_v(init_poses.size());
    for(size_t i=0; i<init_poses.size();i++) mat4_v[i].init_from_cv(init_poses[i]);

    auto depths = cuda_renderer::render(tris, mat4_v, width_local, height_local, proj_mat);
    return f(depths, width_local, height_local, init_poses.size());
}

std::vector<cv::Mat> PoseRefine::render_depth(std::vector<cv::Mat> &init_poses, int down_sample)
{
#ifdef CUDA_ON
    return render_what(cuda_renderer::raw2depth_uint16_cuda, init_poses, down_sample);
#else
    return render_what(cuda_renderer::raw2depth_uint16_cpu, init_poses, down_sample);
#endif
}

std::vector<cv::Mat> PoseRefine::render_mask(std::vector<cv::Mat> &init_poses, int down_sample)
{
#ifdef CUDA_ON
    return render_what(cuda_renderer::raw2mask_uint8_cuda, init_poses, down_sample);
#else
    return render_what(cuda_renderer::raw2mask_uint8_cpu, init_poses, down_sample);
#endif
}

std::vector<std::vector<cv::Mat> > PoseRefine::render_depth_mask(std::vector<cv::Mat> &init_poses, int down_sample)
{
#ifdef CUDA_ON
    return render_what(cuda_renderer::raw2depth_mask_cuda, init_poses, down_sample);
#else
    return render_what(cuda_renderer::raw2depth_mask_cpu, init_poses, down_sample);
#endif
}

cv::Mat so3_map(cv::Mat rot_vec){
    assert(rot_vec.rows == 3 && rot_vec.cols == 1 && rot_vec.type() == CV_32F);

    double theta = cv::norm(rot_vec);
    assert(theta > 1e-6 && "rot vec too small");
    rot_vec /= theta;

    cv::Mat rot_up = cv::Mat(3, 3, CV_32F, cv::Scalar(0));
    rot_up.at<float>(0, 1) = -rot_vec.at<float>(2, 0); rot_up.at<float>(1, 0) = rot_vec.at<float>(2, 0);
    rot_up.at<float>(0, 2) = rot_vec.at<float>(1, 0); rot_up.at<float>(2, 0) = -rot_vec.at<float>(1, 0);
    rot_up.at<float>(1, 2) = -rot_vec.at<float>(0, 0); rot_up.at<float>(2, 1) = rot_vec.at<float>(0, 0);

    cv::Mat R = std::cos(theta)*cv::Mat::eye(3, 3, CV_32F) +
            (1-std::cos(theta))*rot_vec*rot_vec.t() + std::sin(theta) * rot_up;
    return R;
}

std::vector<cv::Mat> PoseRefine::poses_extend(std::vector<cv::Mat> &init_poses, float degree_var)
{
    const bool debug_ = false;
    const bool record_ = false;
    if(debug_){
        if(record_){
            cv::FileStorage fs("/home/meiqua/dump_poses_extend.yml", cv::FileStorage::WRITE);
            fs << "init_poses" << "[";
            for(auto& r: init_poses){
                fs << r;
            }
            fs << "]";
        }
        else{
            cv::FileStorage fs("/home/meiqua/dump_poses_extend.yml", cv::FileStorage::READ);
            cv::FileNode results_fn = fs["init_poses"];
            init_poses.clear();
            init_poses.resize(results_fn.size());
            cv::FileNodeIterator it = results_fn.begin(), it_end = results_fn.end();
            for (int i_r = 0; it != it_end; ++it, ++i_r){
                (*it) >> init_poses[i_r];
            }
        }
    }

    assert(init_poses.size() > 0);
    assert(init_poses[0].type() == CV_32F && init_poses[0].rows == 4 && init_poses[0].cols == 4);

    size_t total_size = 27;
    bool only_2_rot = true;  // 2 rot is neighbor with all 27, but has half the candidate
    if(only_2_rot) total_size = 13;

    std::vector<cv::Mat> extended(init_poses.size() * total_size);
    for(size_t pose_iter=0; pose_iter<init_poses.size(); pose_iter++){
        auto& pose = init_poses[pose_iter];

        size_t shift = 0;
        for(int i=-1; i<=1; i++){
            for(int j=-1; j<=1; j++){
                for(int k=-1; k<=1; k++){

                    if(only_2_rot){
                        if(std::abs(i) + std::abs(j) + std::abs(k) != 2
                                && (std::abs(i) + std::abs(j) + std::abs(k) != 0)) continue;
                    }

                    auto& extended_cur = extended[pose_iter*total_size + shift];
                    extended_cur = pose.clone();
                    if(i==0 && j==0 && k==0){shift++; continue;}

                    float vec_data[3] = {
                        i * degree_var,
                        j * degree_var,
                        k * degree_var
                    };
                    cv::Mat rot_vec(3, 1, CV_32F, vec_data);
                    cv::Mat delta_R = so3_map(rot_vec);


                    cv::Mat R = pose(cv::Rect(0, 0, 3, 3));
                    extended_cur(cv::Rect(0, 0, 3, 3)) = delta_R * R;

                    shift ++;
                }
            }
        }
    }
    return extended;
}

cv::Mat PoseRefine::view_dep(cv::Mat dep)
{
    cv::Mat map = dep;
    double min;
    double max;
    cv::minMaxIdx(map, &min, &max);
    cv::Mat adjMap;
    map.convertTo(adjMap,CV_8UC1, 255 / (max-min), -min);
    cv::Mat falseColorsMap;
    applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_HOT);
    return falseColorsMap;
}


