#include <cuda_kernels.h>
#include <cstdio>

int init_cuda() {
    int selected_device = -1;
    cudaDeviceProp prop;
    int num_devices;
    gpuErrchk(cudaGetDeviceCount(&num_devices));
    printf("NUM OF DEVICES %i\n", num_devices);
    if (num_devices > 0) {
        //set random device!
        //generates a random number between 0 and num_devices to split the threads between GPUs
        selected_device = rand() % num_devices;
        gpuErrchk(cudaSetDevice(selected_device));

        gpuErrchk(cudaGetDeviceProperties(&prop, selected_device));
        //debug code
        if (prop.canMapHostMemory != 1) {
            printf("Device can not map memory.\n");
        }
        printf("L2 Cache size= %u \n", prop.l2CacheSize);
        printf("maxThreadsPerBlock= %u \n", prop.maxThreadsPerBlock);
        printf("maxGridSize= %i \n", prop.maxGridSize[0]);
        printf("sharedMemPerBlock= %lu \n", prop.sharedMemPerBlock);
        printf("deviceOverlap= %i \n", prop.deviceOverlap);
        printf("multiProcessorCount= %i \n", prop.multiProcessorCount);
    }
    printf("selected device: %d\n", selected_device);
    return selected_device;
}

__global__ void cuda_conj_vector(cuComplex* out, cuComplex* in, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        out[i] = cuConjf(in[i]);
    }
}

__global__ void cuda_sincos(cuComplex* out, float phase_inc, float phase, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        float s, c;
        sincosf(phase + i * phase_inc, &s, &c);
        out[i] = make_cuComplex(c, s);
    }
}

__global__ void cuda_mul_vectors(cuComplex* out, cuComplex* in1, cuComplex* in2, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        out[i] = cuCmulf(in1[i], in2[i]);
    }
}

__global__ void cuda_convert_cshort(cuComplex* out, lv_16sc_t* in, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride) {
        int16c_t* iin = (int16c_t *)in;
        out[i] = make_cuComplex(iin[i].r, iin[i].i);
    }
}

__global__ void cuda_max_magt_sq_and_index_stage1(float* magt, int* magt_idx, cuComplex* in, unsigned int size) {
    extern __shared__ float shared[];
    float* max_num = shared;
    int* max_idx = (int*)&max_num[blockDim.x];
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    float magt_sq = 0;
    max_num[tid] = FLT_MIN;

    for(int i = index; i < size; i += stride) {
        magt_sq = cuCrealf(in[i]) * cuCrealf(in[i]) + cuCimagf(in[i]) * cuCimagf(in[i]);
        if(max_num[tid] < magt_sq) {
            max_num[tid] = magt_sq;
            max_idx[tid] = i;
        }
    }
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if(tid < s && index < size) {
            if(max_num[tid] < max_num[tid + s]) {
                max_num[tid] = max_num[tid + s];
                max_idx[tid] = max_idx[tid + s];
            } else if(max_num[tid] == max_num[tid + s]) {
                max_idx[tid] = min(max_idx[tid], max_idx[tid + s]);
            }
        }
        __syncthreads();
    }
    if(tid == 0) {
        magt[blockIdx.x] = max_num[0];
        magt_idx[blockIdx.x] = max_idx[0];
    }
}

__global__ void cuda_max_magt_sq_and_index_stage2(float* magt, int* magt_idx, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if(tid < s && index < size) {
            if(magt[tid] < magt[tid + s]) {
                magt[tid] = magt[tid + s];
                magt_idx[tid] = magt_idx[tid + s];
            } else if(magt[tid] == magt[tid + s]) {
                magt_idx[tid] = min(magt_idx[tid], magt_idx[tid + s]);
            }
        }
        __syncthreads();
    }
}

__global__ void cuda_magt_sq_sum_stage1(float* sum, cuComplex* in, unsigned int size) {
    extern __shared__ float shared[];
    float* num = shared;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;
    float magt_sq = 0;
    num[tid] = 0;

    for(int i = index; i < size; i += stride) {
        magt_sq = cuCrealf(in[i]) * cuCrealf(in[i]) + cuCimagf(in[i]) * cuCimagf(in[i]);
        num[tid] += magt_sq;
    }
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if(tid < s && index < size) {
            num[tid] += num[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0) {
        sum[blockIdx.x] = num[0];
    }
}

__global__ void cuda_magt_sq_sum_stage2(float* sum, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if(tid < s && index < size) {
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }
}

__global__ void cuda_xn_resampler_xn(cuComplex* result, const float* local_code,
                                     float rem_code_phase_chips, float code_phase_step_chips,
                                     float* shifts_chips, unsigned int code_length_chips, int num_out_vectors,
                                     unsigned int num_points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_code_chip_index;
    int current_correlator_tap;
    for (current_correlator_tap = 0; current_correlator_tap < num_out_vectors; current_correlator_tap++)
    {
        for (int n = index; n < num_points; n += stride) {
            // resample code for current tap
            local_code_chip_index = (int)floor(code_phase_step_chips * (float)n + shifts_chips[current_correlator_tap] - rem_code_phase_chips);
            //Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index < 0) local_code_chip_index += (int)code_length_chips * (abs(local_code_chip_index) / code_length_chips + 1);
            local_code_chip_index = local_code_chip_index % code_length_chips;
            result[current_correlator_tap * num_points + n] = make_cuComplex(local_code[local_code_chip_index], 0);
        }
    }
}

__global__ void cuda_x2_dot_prod_xn_stage1(cuComplex* result, const cuComplex* in_common,
                                           cuComplex* phase, cuComplex* in_a, int num_a_vectors,
                                           unsigned int num_points) {
    extern __shared__ cuComplex shared_mem[];
    cuComplex* num = shared_mem;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    for(int n_vec = 0; n_vec < num_a_vectors; n_vec++) {
        num[n_vec * blockDim.x + tid] = make_cuComplex(0, 0);


        for(int i = index; i < num_points; i += stride) {
                num[n_vec * blockDim.x + tid] = cuCaddf(num[n_vec * blockDim.x + tid],
                                                        cuCmulf(cuCmulf(in_common[i], phase[i]),
                                                                in_a[n_vec * num_points + i]));
        }
        __syncthreads();

        for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if(tid < s && index < num_points) {
                num[n_vec * blockDim.x + tid] = cuCaddf(num[n_vec * blockDim.x + tid], num[n_vec * blockDim.x + tid + s]);
            }
            __syncthreads();
        }
        if(tid == 0) {
            result[n_vec * gridDim.x + blockIdx.x] = num[n_vec * blockDim.x];
        }
    }
}

__global__ void cuda_x2_dot_prod_xn_stage2(cuComplex* end_result, cuComplex* result, int num_a_vectors,
                                           unsigned int num_points) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    for(int n_vec = 0; n_vec < num_a_vectors; n_vec++) {
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s && index < num_points) {
                result[n_vec * blockDim.x + tid] = cuCaddf(result[n_vec * blockDim.x + tid], result[n_vec * blockDim.x + tid + s]);
            }
            __syncthreads();
        }
        if(tid == 0) {
            end_result[n_vec] = result[n_vec * blockDim.x];
        }
    }
}


__global__ void cuda_multicorrelator_stage1(cuComplex* result,
                                            float phase_inc,
                                            float phase,
                                            const float* local_code,
                                            float rem_code_phase_chips,
                                            float code_phase_step_chips,
                                            float* shifts_chips,
                                            unsigned int code_length_chips,
                                            const cuComplex* in_common,
                                            int num_out_vectors,
                                            unsigned int num_points) {
    extern __shared__ cuComplex shared_mem[];
    cuComplex* num = shared_mem;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int n = index % num_points;
    int n_vec = index / num_points;
    int tid = threadIdx.x;
    int local_code_chip_index;
    cuComplex phase_out;
    cuComplex resampled;
    float s, c;
    num[tid] = make_cuComplex(0, 0);
    sincosf(phase + n * phase_inc, &s, &c);
    phase_out = make_cuComplex(c, s);
    local_code_chip_index = (int)floor(code_phase_step_chips * (float)n + shifts_chips[n_vec] - rem_code_phase_chips);
    if (local_code_chip_index < 0) {
        local_code_chip_index += (int)code_length_chips * (abs(local_code_chip_index) / code_length_chips + 1);
    }
    local_code_chip_index = local_code_chip_index % code_length_chips;
    resampled = make_cuComplex(local_code[local_code_chip_index], 0);
    num[tid] = cuCaddf(num[tid], cuCmulf(cuCmulf(in_common[n], phase_out), resampled));
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if(tid < s && index < num_points * num_out_vectors) {
            num[tid] = cuCaddf(num[tid], num[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0) {
        result[blockIdx.x] = num[0];
    }
}

__global__ void cuda_multicorrelator_stage2(cuComplex* end_result, cuComplex* result) {
    int tid = threadIdx.x;
    int n_vec = blockIdx.x;
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            result[n_vec * blockDim.x + tid] = cuCaddf(result[n_vec * blockDim.x + tid], result[n_vec * blockDim.x + tid + s]);
        }
        __syncthreads();
    }
    if(tid == 0) {
        end_result[n_vec] = result[n_vec * blockDim.x];
    }
}