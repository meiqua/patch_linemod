#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cpu(cv::Mat& scene_depth_, Mat3x3f& scene_K, Scene_proj_buffer_cpu& scene_buffer){

        cv::Mat scene_depth;
        int depth_type = scene_depth_.type();
        assert(depth_type == CV_16U || depth_type == CV_32S);
        if(depth_type == CV_32S) scene_depth_.convertTo(scene_depth, CV_16U);
        else scene_depth = scene_depth_;

        K = scene_K;
        width = scene_depth.cols;
        height = scene_depth.rows;

        auto& pcd_buffer = scene_buffer.pcd_buffer;
        auto& normal_buffer = scene_buffer.normal_buffer;

        pcd_buffer.clear();
        pcd_buffer.resize(width * height);


        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
            }
        }
        normal_buffer = get_normal(scene_depth, K);


        pcd_ptr = pcd_buffer.data();
        normal_ptr = normal_buffer.data();
}
