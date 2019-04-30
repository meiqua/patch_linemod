#include "pose_tree.h"
#include <queue>
#include <map>
#include <set>
using namespace cv;

template<class T>
static void cpu_exclusive_scan_serial(T* start, uint32_t N){
    T cache = start[0];
    start[0] = 0;
    for (uint32_t i = 1; i < N; i++)
    {
        T temp = cache + start[i-1];
        cache = start[i];
        start[i] = temp;
    }
}

static Vec3f rotate_along_axis(float theta, Vec3f& u3, Vec3f& x3){
    float u = u3[0];
    float v = u3[1];
    float w = u3[2];
    float x = x3[0];
    float y = x3[1];
    float z = x3[2];
    float a = 1-std::cos(theta);
    float b = std::cos(theta);
    float c = std::sin(theta);
    return {
        u*(u*x+v*y+w*z)*a + x*b + (v*z-w*y)*c,
        v*(u*x+v*y+w*z)*a + y*b + (w*x-u*z)*c,
        w*(u*x+v*y+w*z)*a + z*b + (u*y-v*x)*c
    };
}

static cv::Mat pt2view(const Vec3f& pt, float tilt){
    cv::Mat result = cv::Mat::eye(4, 4, CV_32F);

    Vec3f f = -pt/cv::norm(pt);
    Vec3f u = {0, 0, 1};
    Vec3f s = f.cross(u);
    if(cv::countNonZero(s) == 0) s = {1, 0, 0};
    s /= cv::norm(s);
    s = rotate_along_axis(tilt, f, s);
    u = s.cross(f);

    result.at<float>(0, 0) =  s[0]; result.at<float>(0, 1) =  s[1]; result.at<float>(0, 2) =  s[2];
    result.at<float>(1, 0) =  -u[0]; result.at<float>(1, 1) =  -u[1]; result.at<float>(1, 2) =  -u[2];
    result.at<float>(2, 0) = f[0]; result.at<float>(2, 1) = -f[1]; result.at<float>(2, 2) = f[2];

    result.at<float>(0, 3) = -s.dot(pt);
    result.at<float>(1, 3) = u.dot(pt);
    result.at<float>(2, 3) = -f.dot(pt);

    return result;
}

linemodLevelup::Pose_structure hinter_sampling(int level, float radius,
                                               float azimuth_range_min, float azimuth_range_max,
                                               float elev_range_min, float elev_range_max,
                                               float tilt_range_min, float tilt_range_max,
                                               float tilt_step)
{
    float a = 0;
    float b = 1;
    float c = (1 + std::sqrt(5))/2;
    std::vector<Vec3f> pts(12);
    pts[0] = {-b, c, a}; pts[1] = {b, c, a}; pts[ 2] = {-b, -c, a}; pts[ 3] = {b, -c, a};
    pts[4] = {a, -b, c}; pts[5] = {a, b, c}; pts[ 6] = {a, -b, -c}; pts[ 7] = {a, b, -c};
    pts[8] = {c, a, -b}; pts[9] = {c, a, b}; pts[10] = {-c, a, -b}; pts[11] = {-c, a, b};

    std::vector<std::vector<int>> faces = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9},
        {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8}, {3, 9, 4}, {3, 4, 2},
        {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};

    for(int cur_level=1; cur_level<level; cur_level++){
        std::map<std::vector<int>, int> edge_pt_map;
        std::vector<std::vector<int>> faces_new;

        for(const auto& face: faces){
            auto pt_inds = face;
            for(int i=0; i<3; i++){
                std::vector<int> edge = {face[i], face[(i + 1) % 3]};
                int min_edge = (edge[0] < edge[1]) ? edge[0] : edge[1];
                int max_edge = (edge[0] > edge[1]) ? edge[0] : edge[1];
                edge = {min_edge, max_edge};

                if(edge_pt_map.find(edge) == edge_pt_map.end()){
                    int pt_new_id = pts.size();
                    edge_pt_map[edge] = pt_new_id;
                    pt_inds.push_back(pt_new_id);

                    Vec3f pt_new = (pts[edge[0]] + pts[edge[1]])*0.5f;
                    pts.push_back(pt_new);
                }else{
                    pt_inds.push_back(edge_pt_map[edge]);
                }
            }

            faces_new.push_back({pt_inds[0], pt_inds[3], pt_inds[5]});
            faces_new.push_back({pt_inds[3], pt_inds[1], pt_inds[4]});
            faces_new.push_back({pt_inds[3], pt_inds[4], pt_inds[5]});
            faces_new.push_back({pt_inds[5], pt_inds[4], pt_inds[2]});
        }
        faces = faces_new;
    }

    for(auto& pt: pts) pt *= (radius/float(cv::norm(pt)));

    std::vector<std::set<int>> pts_neibor(pts.size());
    for(auto& face: faces){
        pts_neibor[face[0]].insert(face[1]);
        pts_neibor[face[0]].insert(face[2]);
        pts_neibor[face[1]].insert(face[0]);
        pts_neibor[face[1]].insert(face[2]);
        pts_neibor[face[2]].insert(face[0]);
        pts_neibor[face[2]].insert(face[1]);
    }

    int tilt_num = 0;
    for(float tilt_cur = tilt_range_min; tilt_cur<tilt_range_max; tilt_cur+=tilt_step) tilt_num++;

    auto get_unique = [&](int pts_id, int tilt_id){return tilt_id + pts_id*tilt_num;};
    auto is_valid = [&](Vec3f& pt){
        float azimuth = std::atan2(pt[1], pt[0]);
        if(azimuth<0) azimuth += 2*CV_PI;

        Vec2f pt_xy = {pt[0], pt[1]};
        float elev = std::acos(cv::norm(pt_xy)/cv::norm(pt));
        if(pt[2]<0) elev = -elev;

        if(azimuth>=azimuth_range_min && azimuth<=azimuth_range_max &&
                elev>=elev_range_min && elev<=elev_range_max) return true;
        else return false;
    };

    std::vector<int> valid_pts(pts.size(), 0);
    for(size_t i=0; i<pts.size(); i++) if(is_valid(pts[i])) valid_pts[i] = 1;

    auto pts_tight_map = valid_pts;
    int valid_pts_num = valid_pts.back();
    cpu_exclusive_scan_serial(pts_tight_map.data(), pts_tight_map.size());
    valid_pts_num += pts_tight_map.back();

    linemodLevelup::Pose_structure poses;
    poses.Ts.resize(valid_pts_num*tilt_num);
    poses.nodes.resize(valid_pts_num*tilt_num);

    for(size_t i=0; i<pts.size(); i++){
        if(valid_pts[i] == 0) continue;
        int valid_i = pts_tight_map[i];
        for(int tilt_i=0; tilt_i<tilt_num; tilt_i++){
            int cur_id = get_unique(valid_i, tilt_i);
            poses.nodes[cur_id].id = cur_id;
            poses.Ts[cur_id] = pt2view(pts[i], tilt_range_min + tilt_i * tilt_step);

            {  // two tilt neibor
                int neibor_tilt1 = tilt_i - 1;
                if(neibor_tilt1<0) neibor_tilt1 += tilt_num;
                int neibor_tilt2 = tilt_i + 1;
                if(neibor_tilt2>=tilt_num) neibor_tilt2 -= tilt_num;

                int neibor_id1 = get_unique(valid_i, neibor_tilt1);
                int neibor_id2 = get_unique(valid_i, neibor_tilt2);
                poses.nodes[cur_id].adjs.push_back(neibor_id1);
                poses.nodes[cur_id].adjs.push_back(neibor_id2);
            }

            // hinter neibor
            for(auto neibor_pt: pts_neibor[i]){
                int neibor_id = get_unique(pts_tight_map[neibor_pt], tilt_i);
                poses.nodes[cur_id].adjs.push_back(neibor_id);
            }
        }
    }
    return poses;
}
