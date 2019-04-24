#include "icp.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

namespace cuda_icp{

// for debug info
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void transform_pcd_cuda(Vec3f* model_pcd_ptr, uint32_t model_pcd_size, Mat4x4f trans){
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= model_pcd_size) return;

    Vec3f& pcd = model_pcd_ptr[i];
    float new_x = trans[0][0]*pcd.x + trans[0][1]*pcd.y + trans[0][2]*pcd.z + trans[0][3];
    float new_y = trans[1][0]*pcd.x + trans[1][1]*pcd.y + trans[1][2]*pcd.z + trans[1][3];
    float new_z = trans[2][0]*pcd.x + trans[2][1]*pcd.y + trans[2][2]*pcd.z + trans[2][3];
    pcd.x = new_x;
    pcd.y = new_y;
    pcd.z = new_z;
}

template<class Scene>
RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f> &model_pcd, Scene scene,
                                        const ICPConvergenceCriteria criteria){
    RegistrationResult result;
    RegistrationResult backup;

    thrust::host_vector<float> A_host(36, 0);
    thrust::host_vector<float> b_host(6, 0);

    const uint32_t threadsPerBlock = 256;
    const uint32_t numBlocks = (model_pcd.size() + threadsPerBlock - 1)/threadsPerBlock;

    int edge_count = thrust::transform_reduce(thrust::cuda::par.on(cudaStreamPerThread),
                                                model_pcd.begin_thr(), model_pcd.end_thr(),
                                                    thrust__v3f2int(), int(0), thrust::plus<int>());
    cudaStreamSynchronize(cudaStreamPerThread);
    scene.edge_weight = float(edge_count)/model_pcd.size();

    for(uint32_t iter=0; iter<= criteria.max_iteration_; iter++){

        Vec29f Ab_tight;

        if(iter==0 && scene.use_first){
            scene.set_first();
            Ab_tight = thrust::transform_reduce(thrust::cuda::par.on(cudaStreamPerThread),
                                            model_pcd.begin_thr(), model_pcd.end_thr(),
                                                thrust__pcd2Ab__only_T<Scene>(scene),
                                            Vec29f::Zero(), thrust__plus());
        }
        else{
            scene.reset_first();
            Ab_tight = thrust::transform_reduce(thrust::cuda::par.on(cudaStreamPerThread),
                                            model_pcd.begin_thr(), model_pcd.end_thr(), thrust__pcd2Ab<Scene>(scene),
                                            Vec29f::Zero(), thrust__plus());
        }
        cudaStreamSynchronize(cudaStreamPerThread);
        backup = result;

        float& count = Ab_tight[28];
        float& total_error = Ab_tight[27];
        if(count == 0) return result;  // avoid divid 0

        result.fitness_ = float(count) / model_pcd.size();
        result.inlier_rmse_ = std::sqrt(total_error / count);

        // last extra iter, just compute fitness & mse
        if(iter == criteria.max_iteration_) return result;

        if(std::abs(result.fitness_ - backup.fitness_) < criteria.relative_fitness_ &&
           std::abs(result.inlier_rmse_ - backup.inlier_rmse_) < criteria.relative_rmse_){
            return result;
        }

        for(int i=0; i<6; i++) b_host[i] = Ab_tight[21 + i];

        int shift = 0;
        for(int y=0; y<6; y++){
            for(int x=y; x<6; x++){
                A_host[x + y*6] = Ab_tight[shift];
                A_host[y + x*6] = Ab_tight[shift];
                shift++;
            }
        }

        Mat4x4f extrinsic = eigen_slover_666(A_host.data(), b_host.data());

        transform_pcd_cuda<<<numBlocks, threadsPerBlock>>>(model_pcd.data(), model_pcd.size(), extrinsic);
        cudaStreamSynchronize(cudaStreamPerThread);

        result.transformation_ = extrinsic * result.transformation_;
    }

    // never arrive here
    return result;
}

template RegistrationResult ICP_Point2Plane_cuda(device_vector_holder<Vec3f>&, Scene_projective, const ICPConvergenceCriteria);

template <class T>
__global__ void depth2mask(T* depth, uint32_t* mask, uint32_t width, uint32_t height, uint32_t stride){
    uint32_t x = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x*stride>=width) return;
    if(y*stride>=height) return;

    if(depth[x*stride + y*stride*width] > 0) mask[x + y*width] = 1;
}

static const int BLOCK_DIM_X = 16;
static const int BLOCK_DIM_Y = 16;

template <class T>
__global__ void depth2cloud(T* depth, Vec3f* pcd, uint32_t width, uint32_t height, uint32_t* scan, Mat3x3f K,
                          uint32_t stride, uint32_t tl_x, uint32_t tl_y){
    __shared__ uint32_t block_buffer[BLOCK_DIM_X + 2][BLOCK_DIM_Y + 2]; // +2: tile & border

    uint32_t x = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x*stride>=width) return;
    if(y*stride>=height) return;

    uint32_t index_mask = x + y*width;
    uint32_t idx_depth = x*stride + y*stride*width;

    // load tile & border
    block_buffer[threadIdx.x + 1][threadIdx.y + 1] = depth[idx_depth];
    if(threadIdx.x == 0){
        if(x > 0) block_buffer[threadIdx.x][threadIdx.y + 1] = depth[(x-1)*stride + y*stride*width];
        else block_buffer[threadIdx.x][threadIdx.y + 1] = 0;
    }else if(threadIdx.x == blockDim.x - 1){
        if(x < width/stride - 1)
            block_buffer[threadIdx.x + 2][threadIdx.y + 1] = depth[(x+1)*stride + y*stride*width];
        else block_buffer[threadIdx.x + 2][threadIdx.y + 1] = 0;
    }

    if(threadIdx.y == 0){
        if(y > 0) block_buffer[threadIdx.x + 1][threadIdx.y] = depth[x*stride + (y-1)*stride*width];
        else block_buffer[threadIdx.x + 1][threadIdx.y] = 0;
    }else if(threadIdx.y == blockDim.y - 1){
        if(y < height/stride - 1)
            block_buffer[threadIdx.x + 1][threadIdx.y + 2] = depth[x*stride + (y+1)*stride*width];
        else block_buffer[threadIdx.x + 1][threadIdx.y + 2] = 0;
    }
    __syncthreads();

    if(block_buffer[threadIdx.x + 1][threadIdx.y + 1] <= 0) return;

    float z_pcd = depth[idx_depth]/1000.0f;
    float x_pcd = (x + tl_x - K[0][2])/K[0][0]*z_pcd;
    float y_pcd = (y + tl_y - K[1][2])/K[1][1]*z_pcd;

    if(block_buffer[threadIdx.x + 2][threadIdx.y + 1] <=0 ||
            block_buffer[threadIdx.x][threadIdx.y + 1] <=0 ||
            block_buffer[threadIdx.x + 1][threadIdx.y + 2] <=0 ||
            block_buffer[threadIdx.x + 1][threadIdx.y] <=0){
        z_pcd = -z_pcd; // indicate it's a border
    }

    pcd[scan[index_mask]] = {x_pcd, y_pcd, z_pcd};
}

template <class T>
device_vector_holder<Vec3f> depth2cloud_cuda(T *depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                 uint32_t stride, uint32_t tl_x, uint32_t tl_y)
{
    thrust::device_vector<uint32_t> mask(width*height/stride/stride, 0);
    uint32_t* mask_ptr = thrust::raw_pointer_cast(mask.data());

    const dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks_stride((width/stride + 15)/16, (height/stride + 15)/16);
    depth2mask<<<numBlocks_stride, threadsPerBlock>>>(depth, mask_ptr, width, height, stride);

    // avoid blocking per-thread streams
    cudaStreamSynchronize(cudaStreamPerThread);
//            gpuErrchk(cudaPeekAtLastError());

    // scan to find map: depth idx --> cloud idx
    uint32_t mask_back_temp = mask.back();
    thrust::exclusive_scan(mask.begin(), mask.end(), mask.begin(), 0); // in-place scan
    uint32_t total_pcd_num = mask.back() + mask_back_temp;

    device_vector_holder<Vec3f> cloud(total_pcd_num);
    Vec3f* cloud_ptr = cloud.data();
//    gpuErrchk(cudaPeekAtLastError());

    depth2cloud<<<numBlocks_stride, threadsPerBlock>>>(depth, cloud_ptr, width, height,
                                                 mask_ptr, K, stride, tl_x, tl_y);
    cudaStreamSynchronize(cudaStreamPerThread);
//            gpuErrchk(cudaPeekAtLastError());

    return cloud;
}

template device_vector_holder<Vec3f> depth2cloud_cuda(uint16_t *depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                 uint32_t stride, uint32_t tl_x, uint32_t tl_y);
template device_vector_holder<Vec3f> depth2cloud_cuda(int32_t *depth, uint32_t width, uint32_t height, Mat3x3f& K,
                                 uint32_t stride, uint32_t tl_x, uint32_t tl_y);
}



