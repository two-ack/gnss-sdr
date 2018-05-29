/*!
 * \file cuda_multicorrelator.cc
 * \brief High optimized CUDA vector multiTAP correlator class
 * \authors <ul>
 *          <li> Javier Arribas, 2015. jarribas(at)cttc.es
 *          </ul>
 *
 * Class that implements a high optimized vector multiTAP correlator class for CUDAs
 *
 * -------------------------------------------------------------------------
 *
 * Copyright (C) 2010-2015  (see AUTHORS file for a list of contributors)
 *
 * GNSS-SDR is a software defined Global Navigation
 *          Satellite Systems receiver
 *
 * This file is part of GNSS-SDR.
 *
 * GNSS-SDR is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GNSS-SDR is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNSS-SDR. If not, see <http://www.gnu.org/licenses/>.
 *
 * -------------------------------------------------------------------------
 */

#include "cuda_multicorrelator_real_codes.h"
#include "cuda/cuda_kernels.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <gnuradio/gr_complex.h>


cuda_multicorrelator_real_codes::cuda_multicorrelator_real_codes()
{
    cu_sig_in = nullptr;
    cu_local_code_in = nullptr;
    cu_shifts_chips = nullptr;
    cu_corr_out = nullptr;
    cu_local_codes_resampled = nullptr;
    d_code_length_chips = 0;
    d_n_correlators = 0;
}


cuda_multicorrelator_real_codes::~cuda_multicorrelator_real_codes()
{
    if (cu_local_codes_resampled != nullptr)
    {
        this->free();
    }
}


bool cuda_multicorrelator_real_codes::init(int device, int max_signal_length_samples, int code_length_chips, int n_correlators)
{
    d_n_correlators = n_correlators;
    cu_selected_device = device;
    int max_signal_length_samples_pow2 = (int)pow(2, ceil(log2((double)max_signal_length_samples)));
    gpuErrchk(cudaSetDevice(cu_selected_device));
    cu_num_threads = 1024;
    cu_num_blocks = max_signal_length_samples_pow2 / cu_num_threads;
    cu_mul_num_threads = cu_num_threads;
    cu_mul_num_blocks = cu_num_blocks;
    // ALLOCATE MEMORY FOR INTERNAL vectors
    size_t size = max_signal_length_samples_pow2 * sizeof(cuComplex);
    gpuErrchk(cudaHostAlloc((void**)&cu_sig_in, size, cudaHostAllocMapped | cudaHostAllocWriteCombined));
//    gpuErrchk(cudaMalloc((void**)&cu_sig_in, size));
//    gpuErrchk(cudaMalloc((void**)&cu_corr_out, n_correlators * sizeof(cuComplex)));
    gpuErrchk(cudaMalloc((void**)&cu_local_code_in, code_length_chips * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&cu_shifts_chips, n_correlators * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&cu_red_tmp, n_correlators * cu_mul_num_blocks * sizeof(cuComplex)));
//    d_local_codes_resampled = static_cast<std::complex<float>**>(volk_gnsssdr_malloc(n_correlators * sizeof(std::complex<float>*), volk_gnsssdr_get_alignment()));

//    gpuErrchk(cudaMalloc((void**)&cu_phase, size));
    return true;
}


bool cuda_multicorrelator_real_codes::set_local_code_and_taps(int code_length_chips, const float *local_code_in,
                                                              float *shifts_chips, int n_correlations)
{
//    cu_local_code_in = local_code_in;
    gpuErrchk(cudaMemcpy(cu_local_code_in, local_code_in, sizeof(float) * code_length_chips, cudaMemcpyHostToDevice));
//    d_shifts_chips = shifts_chips;
    gpuErrchk(cudaMemcpy(cu_shifts_chips, shifts_chips, sizeof(float) * n_correlations, cudaMemcpyHostToDevice));
    d_code_length_chips = code_length_chips;
    return true;
}


bool cuda_multicorrelator_real_codes::set_input_output_vectors(gr_complex* corr_out, const gr_complex* sig_in)
{
    // Save CUDA pointers
    d_sig_in = sig_in;
    gpuErrchk(cudaHostGetDevicePointer((void **)&cu_corr_out, corr_out, 0));
    return true;
}


void cuda_multicorrelator_real_codes::update_local_code(int correlator_length_samples, float rem_code_phase_chips, float code_phase_step_chips)
{
    gpuErrchk(cudaSetDevice(cu_selected_device));
    cuda_xn_resampler_xn<<<cu_num_blocks, cu_num_threads>>>(cu_local_codes_resampled,
                                                            cu_local_code_in,
                                                            rem_code_phase_chips,
                                                            code_phase_step_chips,
                                                            cu_shifts_chips,
                                                            d_code_length_chips,
                                                            d_n_correlators,
                                                            correlator_length_samples);
    gpuErrchk(cudaGetLastError());
}


bool cuda_multicorrelator_real_codes::Carrier_wipeoff_multicorrelator_resampler(
        float rem_carrier_phase_in_rad,
        float phase_step_rad,
        float rem_code_phase_chips,
        float code_phase_step_chips,
        int signal_length_samples)
{
    gpuErrchk(cudaSetDevice(cu_selected_device));
    int signal_length_samples_pow2 = (int)pow(2, ceil(log2((double)signal_length_samples)));
    memcpy(cu_sig_in, d_sig_in, sizeof(cuComplex) * signal_length_samples);
    memset(cu_sig_in + signal_length_samples, 0, signal_length_samples_pow2 - signal_length_samples);
    cu_mul_num_blocks = signal_length_samples_pow2 / cu_mul_num_threads;
    cuda_multicorrelator_stage1<<<cu_mul_num_blocks * d_n_correlators, cu_mul_num_threads, cu_mul_num_threads * sizeof(cuComplex)>>>(
            cu_red_tmp, -phase_step_rad, -rem_carrier_phase_in_rad,
            cu_local_code_in, rem_code_phase_chips, code_phase_step_chips, cu_shifts_chips, d_code_length_chips,
            cu_sig_in, d_n_correlators, signal_length_samples_pow2);
    gpuErrchk(cudaGetLastError());
    cuda_multicorrelator_stage2<<<d_n_correlators, cu_mul_num_blocks>>>(cu_corr_out, cu_red_tmp);
    gpuErrchk(cudaGetLastError());
//    gpuErrchk(cudaMemcpy(d_corr_out, cu_corr_out, sizeof(cuComplex) * d_n_correlators, cudaMemcpyDeviceToHost));
    return true;
}


bool cuda_multicorrelator_real_codes::free()
{
    gpuErrchk(cudaSetDevice(cu_selected_device));
    gpuErrchk(cudaFree(cu_red_tmp));
    gpuErrchk(cudaFreeHost(cu_sig_in));
//    gpuErrchk(cudaMalloc((void**)&cu_sig_in, size));
//    gpuErrchk(cudaFree(cu_corr_out));
    gpuErrchk(cudaFree(cu_local_code_in));
    gpuErrchk(cudaFree(cu_shifts_chips));
    return true;
}
