#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cpu(cv::Mat& scene_depth_, cv::Mat& depth_edge,
                                                 Mat3x3f& scene_K, Scene_proj_buffer_cpu& scene_buffer){

        cv::Mat scene_depth;
        int depth_type = scene_depth_.type();
        assert(depth_type == CV_16U || depth_type == CV_32S);
        if(depth_type == CV_32S) scene_depth_.convertTo(scene_depth, CV_16U);
        else scene_depth = scene_depth_;

        K = scene_K;
        width = scene_depth.cols;
        height = scene_depth.rows;

        auto& pcd_buffer = scene_buffer.pcd_buffer;
        auto& normal_buffer = scene_buffer.pcd_buffer;
        auto& edge_buffer = scene_buffer.edge_buffer;

        pcd_buffer.clear();
        pcd_buffer.resize(width * height);

        edge_buffer.clear();
        edge_buffer.resize(width * height, -1);

        for(int r=0; r<height; r++){
            for(int c=0; c<width; c++){
                pcd_buffer[c + r*width] = dep2pcd(c, r, scene_depth.at<uint16_t>(r, c), K);
            }
        }
        normal_buffer = get_normal(scene_depth, K);

        int kernel_size = 2;
        cv::Mat edge_idx_twist(depth_edge.size(), CV_32S);  // avoid falling to background when query
        for(int r=0+kernel_size; r<depth_edge.rows - kernel_size; r++){
            for(int c=0+kernel_size; c<depth_edge.cols - kernel_size; c++){

                if(depth_edge.at<uchar>(r, c) > 0){

                    int real_idx = -1;
                    int max_depth = 0;
                    for(int i=-kernel_size; i<=kernel_size; i++){
                        for(int j=-kernel_size; j<=kernel_size; j++){

                            int new_r = r + i;
                            int new_c = c + j;
                            if(scene_depth.at<uint16_t>(new_r, new_c) > max_depth){
                                max_depth = scene_depth.at<uint16_t>(new_r, new_c);
                                real_idx = new_c + new_r*depth_edge.cols;
                            }
                        }
                    }
                    edge_idx_twist.at<int>(r, c) = real_idx;
                }
            }
        }


        cv::Mat dist_buffer(depth_edge.size(), CV_32FC1, FLT_MAX);
        kernel_size = int(max_dist_diff+0.5f);
        for(int r=0+kernel_size; r<depth_edge.rows - kernel_size; r++){
            for(int c=0+kernel_size; c<depth_edge.cols - kernel_size; c++){

                if(depth_edge.at<uchar>(r, c) > 0){

                    for(int i=-kernel_size; i<=kernel_size; i++){
                        for(int j=-kernel_size; j<=kernel_size; j++){

                            float dist_sq = pow2(i) + pow2(j);

                            // don't go too far
                            if(dist_sq > pow2(max_edge_diff)) continue;

                            int new_r = r + i;
                            int new_c = c + j;

                            // if closer
                            if(dist_sq < dist_buffer.at<float>(new_r, new_c)){
                                edge_buffer[new_c + new_r*depth_edge.cols] = edge_idx_twist.at<int>(r, c);
                                dist_buffer.at<float>(new_r, new_c) = dist_sq;
                            }
                        }
                    }
                }
            }
        }

        pcd_ptr = pcd_buffer.data();
        normal_ptr = normal_buffer.data();
        edge_ptr = edge_buffer.data();
}
