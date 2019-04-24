#include "depth_scene.h"

void Scene_projective::init_Scene_projective_cuda(cv::Mat &scene_depth, Mat3x3f &scene_K,
                                                   Scene_proj_buffer_cuda& scene_buffer){
    Scene_proj_buffer_cpu scene_buffer_cpu;
    init_Scene_projective_cpu(scene_depth, scene_K, scene_buffer_cpu);

    scene_buffer.pcd_buffer.__malloc(scene_buffer_cpu.pcd_buffer.size());
    thrust::copy(scene_buffer_cpu.pcd_buffer.begin(), scene_buffer_cpu.pcd_buffer.end(),
                 scene_buffer.pcd_buffer.begin_thr());

    scene_buffer.normal_buffer.__malloc(scene_buffer_cpu.normal_buffer.size());
    thrust::copy(scene_buffer_cpu.normal_buffer.begin(), scene_buffer_cpu.normal_buffer.end(),
                 scene_buffer.normal_buffer.begin_thr());

    pcd_ptr = scene_buffer.pcd_buffer.data();
    normal_ptr = scene_buffer.normal_buffer.data();
}
