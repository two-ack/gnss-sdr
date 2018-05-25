/*!
 * \file pcps_cuda_acquisition.cc
 * \brief This class implements a Parallel Code Phase Search Acquisition
 * using CUDA to offload some functions to the GPU.
 *
 *  Acquisition strategy (Kay Borre book + CFAR threshold).
 *  <ol>
 *  <li> Compute the input signal power estimation
 *  <li> Doppler serial search loop
 *  <li> Perform the FFT-based circular convolution (parallel time search)
 *  <li> Record the maximum peak and the associated synchronization parameters
 *  <li> Compute the test statistics and compare to the threshold
 *  <li> Declare positive or negative acquisition using a message port
 *  </ol>
 *
 * Kay Borre book: K.Borre, D.M.Akos, N.Bertelsen, P.Rinder, and S.H.Jensen,
 * "A Software-Defined GPS and Galileo Receiver. A Single-Frequency
 * Approach", Birkhauser, 2007. pp 81-84
 *
 * \authors <ul>
 *          <li> Javier Arribas, 2011. jarribas(at)cttc.es
 *          <li> Luis Esteve, 2012. luis(at)epsilon-formacion.com
 *          <li> Marc Molina, 2013. marc.molina.pena@gmail.com
 *          </ul>
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

#include "pcps_cuda_acquisition.h"
#include "GPS_L1_CA.h"         // for GPS_TWO_PI
#include "GLONASS_L1_L2_CA.h"  // for GLONASS_TWO_PI"
#include "cuda/cuda_kernels.h"
#include <glog/logging.h>
#include <gnuradio/io_signature.h>
#include <matio.h>
#include <volk/volk.h>
#include <cstring>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

void print_buff(cuComplex* buff, unsigned int size, const char* msg) {
    gr_complex* tmp = (gr_complex*)malloc(sizeof(gr_complex) * size);
    gpuErrchk(cudaMemcpy(tmp, buff, sizeof(gr_complex) * size, cudaMemcpyDeviceToHost));
    std::cout << msg << " ";
    for(int i = 0; i < size; i++) {
        std::cout << "(" << tmp[i].real() << "; " << tmp[i].imag() << ") ";
    }
    std::cout << std::endl;
    free(tmp);
}

using google::LogMessage;

pcps_cuda_acquisition_sptr pcps_make_cuda_acquisition(pcps_cuda_conf_t conf_)
{
    return pcps_cuda_acquisition_sptr(new pcps_cuda_acquisition(conf_));
}

pcps_cuda_acquisition::pcps_cuda_acquisition(pcps_cuda_conf_t conf_) : gr::block("pcps_cuda_acquisition",
                                               gr::io_signature::make(1, 1, conf_.it_size * conf_.sampled_ms * conf_.samples_per_ms* (conf_.bit_transition_flag ? 2 : 1)),
                                               gr::io_signature::make(0, 0, conf_.it_size * conf_.sampled_ms * conf_.samples_per_ms* (conf_.bit_transition_flag ? 2 : 1)))
{
    this->message_port_register_out(pmt::mp("events"));
    acq_parameters = conf_;
    d_sample_counter = 0;  // SAMPLE COUNTER
    d_active = false;
    d_state = 0;
    d_old_freq = conf_.freq;
    d_well_count = 0;
    d_fft_size = acq_parameters.sampled_ms * acq_parameters.samples_per_ms;
    d_mag = 0;
    d_input_power = 0.0;
    d_num_doppler_bins = 0;
    d_threshold = 0.0;
    d_doppler_step = 0;
    d_doppler_center_step_two = 0.0;
    d_test_statistics = 0.0;
    d_channel = 0;
    cu_data_buffer = nullptr;
    cu_data_buffer_sc = nullptr;
    cu_buffer_fft_codes = nullptr;
    cu_fft_buffer = nullptr;
    cu_ifft_buffer = nullptr;
    cu_grid_doppler_wipeoffs = nullptr;
    cu_grid_doppler_wipeoffs_step_two = nullptr;
    cu_magt_tmp = nullptr;
    cu_magt_idx_tmp = nullptr;
    cu_fft_plan = -1;
    if (conf_.it_size == sizeof(gr_complex))
    {
        d_cshort = false;
    }
    else
    {
        d_cshort = true;
    }

    // COD:
    // Experimenting with the overlap/save technique for handling bit trannsitions
    // The problem: Circular correlation is asynchronous with the received code.
    // In effect the first code phase used in the correlation is the current
    // estimate of the code phase at the start of the input buffer. If this is 1/2
    // of the code period a bit transition would move all the signal energy into
    // adjacent frequency bands at +/- 1/T where T is the integration time.
    //
    // We can avoid this by doing linear correlation, effectively doubling the
    // size of the input buffer and padding the code with zeros.
    if (acq_parameters.bit_transition_flag)
    {
        d_fft_size *= 2;
        acq_parameters.max_dwells = 1;  //Activation of acq_parameters.bit_transition_flag invalidates the value of acq_parameters.max_dwells
    }

    d_fft_size_pow2 = (int)pow(2, ceil(log2((double)d_fft_size)));
    d_fft_padding_len = d_fft_size_pow2 - d_fft_size;

    d_gnss_synchro = 0;
    d_worker_active = false;
    grid_ = arma::fmat();
    d_step_two = false;
}


pcps_cuda_acquisition::~pcps_cuda_acquisition()
{
    gpuErrchk(cudaSetDevice(selected_device));
    if (d_num_doppler_bins > 0)
    {
        for (unsigned int i = 0; i < d_num_doppler_bins; i++)
        {
            gpuErrchk(cudaFree(cu_grid_doppler_wipeoffs[i]));
        }
        free(cu_grid_doppler_wipeoffs);
    }
    if (acq_parameters.make_2_steps)
    {
        for (unsigned int i = 0; i < acq_parameters.num_doppler_bins_step2; i++)
        {
            gpuErrchk(cudaFree(cu_grid_doppler_wipeoffs_step_two[i]));
        }
        free(cu_grid_doppler_wipeoffs_step_two);
    }
    gpuErrchk(cudaFree(cu_buffer_fft_codes));
    gpuErrchk(cudaFree(cu_fft_buffer));
    gpuErrchk(cudaFree(cu_ifft_buffer));
    gpuErrchk(cudaFree(cu_data_buffer));
    gpuErrchk(cudaFree(cu_magt_tmp));
    gpuErrchk(cudaFree(cu_magt_idx_tmp));
    if (d_cshort)
    {
        gpuErrchk(cudaFree(cu_data_buffer_sc));
    }
    if(cu_fft_plan != -1) {
        cufftErrchk(cufftDestroy(cu_fft_plan));
    }
}

void pcps_cuda_acquisition::set_local_code(std::complex<float> *code)
{
    gpuErrchk(cudaSetDevice(selected_device));
    // reset the intermediate frequency
    acq_parameters.freq = d_old_freq;
    // This will check if it's fdma, if yes will update the intermediate frequency and the doppler grid
    if (is_fdma())
    {
        update_grid_doppler_wipeoffs();
    }
    // COD
    // Here we want to create a buffer that looks like this:
    // [ 0 0 0 ... 0 c_0 c_1 ... c_L]
    // where c_i is the local code and there are L zeros and L chips
    gr::thread::scoped_lock lock(d_setlock);  // require mutex with work function called by the scheduler
    if (acq_parameters.bit_transition_flag)
    {
        int offset = d_fft_size / 2;
        gpuErrchk(cudaMemset(cu_fft_buffer, 0, sizeof(cuComplex) * offset));
        gpuErrchk(cudaMemcpy(cu_fft_buffer + offset, code, sizeof(cuComplex) * offset, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(cu_fft_buffer + d_fft_size, 0, sizeof(cuComplex) * d_fft_padding_len));
    }
    else
    {
        gpuErrchk(cudaMemcpy(cu_fft_buffer, code, sizeof(cuComplex) * d_fft_size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemset(cu_fft_buffer + d_fft_size, 0, sizeof(cuComplex) * d_fft_padding_len));
    }

    cufftErrchk(cufftExecC2C(cu_fft_plan, cu_fft_buffer, cu_fft_buffer, CUFFT_FORWARD));
    cuda_conj_vector<<<cu_num_blocks, cu_num_threads>>>(cu_buffer_fft_codes, cu_fft_buffer, d_fft_size_pow2);
    gpuErrchk(cudaGetLastError());
}

bool pcps_cuda_acquisition::is_fdma()
{
    // Dealing with FDMA system
    if (strcmp(d_gnss_synchro->Signal, "1G") == 0)
    {
        acq_parameters.freq += DFRQ1_GLO * GLONASS_PRN.at(d_gnss_synchro->PRN);
        LOG(INFO) << "Trying to acquire SV PRN " << d_gnss_synchro->PRN << " with freq " << acq_parameters.freq << " in Glonass Channel " << GLONASS_PRN.at(d_gnss_synchro->PRN) << std::endl;
        return true;
    }
    else if (strcmp(d_gnss_synchro->Signal, "2G") == 0)
    {
        acq_parameters.freq += DFRQ2_GLO * GLONASS_PRN.at(d_gnss_synchro->PRN);
        LOG(INFO) << "Trying to acquire SV PRN " << d_gnss_synchro->PRN << " with freq " << acq_parameters.freq << " in Glonass Channel " << GLONASS_PRN.at(d_gnss_synchro->PRN) << std::endl;
        return true;
    }
    else
    {
        return false;
    }
}

void pcps_cuda_acquisition::update_local_carrier(cuComplex* carrier_vector, int correlator_length_samples, float freq)
{
    gpuErrchk(cudaSetDevice(selected_device));
    float phase_step_rad = GPS_TWO_PI * freq / static_cast<float>(acq_parameters.fs_in);
    cuda_sincos<<<cu_num_blocks, cu_num_threads>>>(carrier_vector, -phase_step_rad, 0, correlator_length_samples);
    gpuErrchk(cudaGetLastError());
}

void pcps_cuda_acquisition::init()
{
    d_gnss_synchro->Flag_valid_acquisition = false;
    d_gnss_synchro->Flag_valid_symbol_output = false;
    d_gnss_synchro->Flag_valid_pseudorange = false;
    d_gnss_synchro->Flag_valid_word = false;

    d_gnss_synchro->Acq_delay_samples = 0.0;
    d_gnss_synchro->Acq_doppler_hz = 0.0;
    d_gnss_synchro->Acq_samplestamp_samples = 0;
    d_mag = 0.0;
    d_input_power = 0.0;

    d_num_doppler_bins = static_cast<unsigned int>(std::ceil(static_cast<double>(static_cast<int>(acq_parameters.doppler_max) - static_cast<int>(-acq_parameters.doppler_max)) / static_cast<double>(d_doppler_step)));


    int err = init_cu();
    if(err != 0) {
        DLOG(ERROR) << "Unable to initialize cuda";
        exit(err);
    }
    d_worker_active = false;
    if (acq_parameters.dump)
    {
        unsigned int effective_fft_size = (acq_parameters.bit_transition_flag ? (d_fft_size / 2) : d_fft_size);
        grid_ = arma::fmat(effective_fft_size, d_num_doppler_bins, arma::fill::zeros);
    }
}

int pcps_cuda_acquisition::init_cu()
{
    selected_device = init_cuda();
    if(selected_device == -1) {
        return -1;
    }

    cu_num_threads = 1024;
    cu_num_blocks = d_fft_size_pow2 / cu_num_threads;
    cu_magt_blocks = cu_num_blocks;

    gpuErrchk(cudaSetDevice(selected_device));

    cufftErrchk(cufftPlan1d(&cu_fft_plan, d_fft_size, CUFFT_C2C, 1));
    cufftErrchk(cufftSetStream(cu_fft_plan, cudaStreamPerThread));

    gpuErrchk(cudaMalloc((void **)&cu_data_buffer, sizeof(cuComplex) * d_fft_size_pow2));
    if (d_cshort)
    {
        gpuErrchk(cudaMalloc((void **)&cu_data_buffer_sc, sizeof(lv_16sc_t) * d_fft_size_pow2));
    }

    gpuErrchk(cudaMalloc((void **)&cu_fft_buffer, sizeof(cuComplex) * d_fft_size_pow2));
    gpuErrchk(cudaMalloc((void **)&cu_ifft_buffer, sizeof(cuComplex) * d_fft_size_pow2));
    gpuErrchk(cudaMalloc((void **)&cu_buffer_fft_codes, sizeof(cuComplex) * d_fft_size_pow2));
    gpuErrchk(cudaMalloc((void **)&cu_magt_tmp, sizeof(float) * cu_magt_blocks));
    gpuErrchk(cudaMalloc((void **)&cu_magt_idx_tmp, sizeof(float) * cu_magt_blocks));

    // Create the carrier Doppler wipeoff signals
    cu_grid_doppler_wipeoffs = (cuComplex**)malloc(d_num_doppler_bins * sizeof(cuComplex*));

    if (acq_parameters.make_2_steps)
    {
        cu_grid_doppler_wipeoffs_step_two = (cuComplex**)malloc(acq_parameters.num_doppler_bins_step2 * sizeof(cuComplex*));
        for (unsigned int doppler_index = 0; doppler_index < acq_parameters.num_doppler_bins_step2; doppler_index++)
        {
            gpuErrchk(cudaMalloc((void**)&cu_grid_doppler_wipeoffs_step_two[doppler_index], sizeof(cuComplex) * d_fft_size_pow2));
            gpuErrchk(cudaMemset(cu_grid_doppler_wipeoffs_step_two[doppler_index] + d_fft_size, 0, sizeof(cuComplex) * d_fft_padding_len));
        }
    }

    for (unsigned int doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++)
    {
        gpuErrchk(cudaMalloc((void**)&cu_grid_doppler_wipeoffs[doppler_index], sizeof(cuComplex) * d_fft_size_pow2));
        gpuErrchk(cudaMemset(cu_grid_doppler_wipeoffs[doppler_index] + d_fft_size, 0, sizeof(cuComplex) * d_fft_padding_len));
        int doppler = -static_cast<int>(acq_parameters.doppler_max) + d_doppler_step * doppler_index;
        update_local_carrier(cu_grid_doppler_wipeoffs[doppler_index], d_fft_size, acq_parameters.freq + doppler);
    }
    return 0;
}

void pcps_cuda_acquisition::update_grid_doppler_wipeoffs()
{
    for (unsigned int doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++)
    {
        int doppler = -static_cast<int>(acq_parameters.doppler_max) + d_doppler_step * doppler_index;
        update_local_carrier(cu_grid_doppler_wipeoffs[doppler_index], d_fft_size, acq_parameters.freq + doppler);
    }
}

void pcps_cuda_acquisition::update_grid_doppler_wipeoffs_step2()
{
    for (unsigned int doppler_index = 0; doppler_index < acq_parameters.num_doppler_bins_step2; doppler_index++)
    {
        float doppler = (static_cast<float>(doppler_index) - static_cast<float>(acq_parameters.num_doppler_bins_step2) / 2.0) * acq_parameters.doppler_step2;
        update_local_carrier(cu_grid_doppler_wipeoffs_step_two[doppler_index], d_fft_size, d_doppler_center_step_two + doppler);
    }
}

void pcps_cuda_acquisition::set_state(int state)
{
    gr::thread::scoped_lock lock(d_setlock);  // require mutex with work function called by the scheduler
    d_state = state;
    if (d_state == 1)
    {
        d_gnss_synchro->Acq_delay_samples = 0.0;
        d_gnss_synchro->Acq_doppler_hz = 0.0;
        d_gnss_synchro->Acq_samplestamp_samples = 0;
        d_well_count = 0;
        d_mag = 0.0;
        d_input_power = 0.0;
        d_test_statistics = 0.0;
        d_active = true;
    }
    else if (d_state == 0)
    {
    }
    else
    {
        LOG(ERROR) << "State can only be set to 0 or 1";
    }
}

void pcps_cuda_acquisition::send_positive_acquisition()
{
    // 6.1- Declare positive acquisition using a message port
    //0=STOP_CHANNEL 1=ACQ_SUCCEES 2=ACQ_FAIL
    DLOG(INFO) << "positive acquisition"
               << ", satellite " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
               << ", sample_stamp " << d_sample_counter
               << ", test statistics value " << d_test_statistics
               << ", test statistics threshold " << d_threshold
               << ", code phase " << d_gnss_synchro->Acq_delay_samples
               << ", doppler " << d_gnss_synchro->Acq_doppler_hz
               << ", magnitude " << d_mag
               << ", input signal power " << d_input_power;

    this->message_port_pub(pmt::mp("events"), pmt::from_long(1));
}


void pcps_cuda_acquisition::send_negative_acquisition()
{
    // 6.2- Declare negative acquisition using a message port
    //0=STOP_CHANNEL 1=ACQ_SUCCEES 2=ACQ_FAIL
    DLOG(INFO) << "negative acquisition"
               << ", satellite " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
               << ", sample_stamp " << d_sample_counter
               << ", test statistics value " << d_test_statistics
               << ", test statistics threshold " << d_threshold
               << ", code phase " << d_gnss_synchro->Acq_delay_samples
               << ", doppler " << d_gnss_synchro->Acq_doppler_hz
               << ", magnitude " << d_mag
               << ", input signal power " << d_input_power;

    this->message_port_pub(pmt::mp("events"), pmt::from_long(2));
}

void pcps_cuda_acquisition::acquisition_core(unsigned long int samp_count)
{
    gpuErrchk(cudaSetDevice(selected_device));
    gr::thread::scoped_lock lk(d_setlock);

    // initialize acquisition algorithm
    uint32_t indext = 0;
    float magt = 0.0;
    cuComplex* in = cu_data_buffer;  //Get the input samples pointer
    int effective_fft_size = (acq_parameters.bit_transition_flag ? d_fft_size / 2 : d_fft_size);
    int effective_fft_size_pow2 = (int)pow(2, ceil(log2((double)effective_fft_size)));
    if (d_cshort)
    {
        cuda_convert_cshort<<<cu_num_blocks, cu_num_threads>>>(cu_data_buffer, cu_data_buffer_sc, d_fft_size_pow2);
    }
    float fft_normalization_factor = static_cast<float>(d_fft_size) * static_cast<float>(d_fft_size);

    d_input_power = 0.0;
    d_mag = 0.0;
    d_well_count++;

    DLOG(INFO) << "Channel: " << d_channel
               << " , doing acquisition of satellite: " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
               << " ,sample stamp: " << samp_count << ", threshold: "
               << d_threshold << ", doppler_max: " << acq_parameters.doppler_max
               << ", doppler_step: " << d_doppler_step
               << ", use_CFAR_algorithm_flag: " << (acq_parameters.use_CFAR_algorithm_flag ? "true" : "false");

    lk.unlock();
    if (acq_parameters.use_CFAR_algorithm_flag)
    {
        // 1- (optional) Compute the input signal power estimation
        cuda_magt_sq_sum(&d_input_power, in, d_fft_size_pow2);
        d_input_power /= static_cast<float>(d_fft_size);
    }
    // 2- Doppler frequency search loop
    if (!d_step_two)
    {
        for (unsigned int doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++)
        {
            // doppler search steps
            int doppler = -static_cast<int>(acq_parameters.doppler_max) + d_doppler_step * doppler_index;

            cuda_mul_vectors<<<cu_num_blocks, cu_num_threads>>>(cu_fft_buffer, in, cu_grid_doppler_wipeoffs[doppler_index], d_fft_size_pow2);
            gpuErrchk(cudaGetLastError());
            // 3- Perform the FFT-based convolution  (parallel time search)
            // Compute the FFT of the carrier wiped--off incoming signal
            cufftErrchk(cufftExecC2C(cu_fft_plan, cu_fft_buffer, cu_ifft_buffer, CUFFT_FORWARD));

            // Multiply carrier wiped--off, Fourier transformed incoming signal
            // with the local FFT'd code reference using SIMD operations with VOLK library
            cuda_mul_vectors<<<cu_num_blocks, cu_num_threads>>>(cu_ifft_buffer, cu_ifft_buffer, cu_buffer_fft_codes, d_fft_size_pow2);
            gpuErrchk(cudaGetLastError());
            // compute the inverse FFT
            cufftErrchk(cufftExecC2C(cu_fft_plan, cu_ifft_buffer, cu_ifft_buffer, CUFFT_INVERSE));

            // Search maximum
            int offset = (acq_parameters.bit_transition_flag ? effective_fft_size : 0);
            cuda_max_magt_sq_and_index(&magt, (int*)&indext, cu_ifft_buffer + offset, effective_fft_size_pow2);

            if (acq_parameters.use_CFAR_algorithm_flag)
            {
                // Normalize the maximum value to correct the scale factor introduced by FFTW
                magt /= (fft_normalization_factor * fft_normalization_factor);
            }
            // 4- record the maximum peak and the associated synchronization parameters
            if (d_mag < magt)
            {
                d_mag = magt;

                if (!acq_parameters.use_CFAR_algorithm_flag)
                {
                    cuda_magt_sq_sum(&d_input_power, cu_ifft_buffer + offset, effective_fft_size_pow2);
                    d_input_power = (d_input_power - d_mag) / (effective_fft_size - 1);
                }

                // In case that acq_parameters.bit_transition_flag = true, we compare the potentially
                // new maximum test statistics (d_mag/d_input_power) with the value in
                // d_test_statistics. When the second dwell is being processed, the value
                // of d_mag/d_input_power could be lower than d_test_statistics (i.e,
                // the maximum test statistics in the previous dwell is greater than
                // current d_mag/d_input_power). Note that d_test_statistics is not
                // restarted between consecutive dwells in multidwell operation.

                if (d_test_statistics < (d_mag / d_input_power) or !acq_parameters.bit_transition_flag)
                {
                    d_gnss_synchro->Acq_delay_samples = static_cast<double>(indext % acq_parameters.samples_per_code);
                    d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
                    d_gnss_synchro->Acq_samplestamp_samples = samp_count;

                    // 5- Compute the test statistics and compare to the threshold
                    //d_test_statistics = 2 * d_fft_size * d_mag / d_input_power;
                    d_test_statistics = d_mag / d_input_power;
                }
            }
            // Record results to file if required
            if (acq_parameters.dump)
            {
                gr_complex* tmp = (gr_complex*)malloc(sizeof(gr_complex) * effective_fft_size);
                gpuErrchk(cudaMemcpy(tmp, cu_ifft_buffer + offset, sizeof(gr_complex) * effective_fft_size, cudaMemcpyDeviceToHost));
                volk_32fc_magnitude_squared_32f(grid_.colptr(doppler_index), tmp, effective_fft_size);
                free(tmp);
                if (doppler_index == (d_num_doppler_bins - 1))
                {
                    std::string filename = acq_parameters.dump_filename;
                    filename.append("_");
                    filename.append(1, d_gnss_synchro->System);
                    filename.append("_");
                    filename.append(1, d_gnss_synchro->Signal[0]);
                    filename.append(1, d_gnss_synchro->Signal[1]);
                    filename.append("_sat_");
                    filename.append(std::to_string(d_gnss_synchro->PRN));
                    filename.append(".mat");
                    mat_t* matfp = Mat_CreateVer(filename.c_str(), NULL, MAT_FT_MAT73);
                    if (matfp == NULL)
                    {
                        std::cout << "Unable to create or open Acquisition dump file" << std::endl;
                        acq_parameters.dump = false;
                    }
                    else
                    {
                        size_t dims[2] = {static_cast<size_t>(effective_fft_size), static_cast<size_t>(d_num_doppler_bins)};
                        matvar_t* matvar = Mat_VarCreate("grid", MAT_C_SINGLE, MAT_T_SINGLE, 2, dims, grid_.memptr(), 0);
                        Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
                        Mat_VarFree(matvar);

                        dims[0] = static_cast<size_t>(1);
                        dims[1] = static_cast<size_t>(1);
                        matvar = Mat_VarCreate("doppler_max", MAT_C_SINGLE, MAT_T_UINT32, 1, dims, &acq_parameters.doppler_max, 0);
                        Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
                        Mat_VarFree(matvar);

                        matvar = Mat_VarCreate("doppler_step", MAT_C_SINGLE, MAT_T_UINT32, 1, dims, &d_doppler_step, 0);
                        Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_ZLIB);  // or MAT_COMPRESSION_NONE
                        Mat_VarFree(matvar);

                        Mat_Close(matfp);
                    }
                }
            }
        }
    }
    else
    {
        for (unsigned int doppler_index = 0; doppler_index < acq_parameters.num_doppler_bins_step2; doppler_index++)
        {
            // doppler search steps
            float doppler = d_doppler_center_step_two + (static_cast<float>(doppler_index) - static_cast<float>(acq_parameters.num_doppler_bins_step2) / 2.0) * acq_parameters.doppler_step2;

            cuda_mul_vectors<<<cu_num_blocks, cu_num_threads>>>(cu_fft_buffer, in, cu_grid_doppler_wipeoffs_step_two[doppler_index], d_fft_size_pow2);
            gpuErrchk(cudaGetLastError());

            // 3- Perform the FFT-based convolution  (parallel time search)
            // Compute the FFT of the carrier wiped--off incoming signal
            cufftErrchk(cufftExecC2C(cu_fft_plan, cu_fft_buffer, cu_ifft_buffer, CUFFT_FORWARD));

            // Multiply carrier wiped--off, Fourier transformed incoming signal
            // with the local FFT'd code reference using SIMD operations with VOLK library
            cuda_mul_vectors<<<cu_num_blocks, cu_num_threads>>>(cu_ifft_buffer, cu_ifft_buffer, cu_buffer_fft_codes, d_fft_size_pow2);
            gpuErrchk(cudaGetLastError());

            // compute the inverse FFT
            cufftErrchk(cufftExecC2C(cu_fft_plan, cu_ifft_buffer, cu_ifft_buffer, CUFFT_INVERSE));

            // Search maximum
            int offset = (acq_parameters.bit_transition_flag ? effective_fft_size : 0);
            cuda_max_magt_sq_and_index(&magt, (int*)&indext, cu_ifft_buffer + offset, effective_fft_size_pow2);

            if (acq_parameters.use_CFAR_algorithm_flag)
            {
                // Normalize the maximum value to correct the scale factor introduced by FFTW
                magt /= (fft_normalization_factor * fft_normalization_factor);
            }
            // 4- record the maximum peak and the associated synchronization parameters
            if (d_mag < magt)
            {
                d_mag = magt;

                if (!acq_parameters.use_CFAR_algorithm_flag)
                {
                    // Search grid noise floor approximation for this doppler line
                    cuda_magt_sq_sum(&d_input_power, cu_ifft_buffer + offset, effective_fft_size_pow2);
                    d_input_power = (d_input_power - d_mag) / (effective_fft_size - 1);
                }

                // In case that acq_parameters.bit_transition_flag = true, we compare the potentially
                // new maximum test statistics (d_mag/d_input_power) with the value in
                // d_test_statistics. When the second dwell is being processed, the value
                // of d_mag/d_input_power could be lower than d_test_statistics (i.e,
                // the maximum test statistics in the previous dwell is greater than
                // current d_mag/d_input_power). Note that d_test_statistics is not
                // restarted between consecutive dwells in multidwell operation.

                if (d_test_statistics < (d_mag / d_input_power) or !acq_parameters.bit_transition_flag)
                {
                    d_gnss_synchro->Acq_delay_samples = static_cast<double>(indext % acq_parameters.samples_per_code);
                    d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
                    d_gnss_synchro->Acq_samplestamp_samples = samp_count;

                    // 5- Compute the test statistics and compare to the threshold
                    //d_test_statistics = 2 * d_fft_size * d_mag / d_input_power;
                    d_test_statistics = d_mag / d_input_power;
                }
            }
        }
    }
    lk.lock();
    if (!acq_parameters.bit_transition_flag)
    {
        if (d_test_statistics > d_threshold)
        {
            d_active = false;
            if (acq_parameters.make_2_steps)
            {
                if (d_step_two)
                {
                    send_positive_acquisition();
                    d_step_two = false;
                    d_state = 0;  // Positive acquisition
                }
                else
                {
                    d_step_two = true;  // Clear input buffer and make small grid acquisition
                    d_state = 0;
                }
            }
            else
            {
                send_positive_acquisition();
                d_state = 0;  // Positive acquisition
            }
        }
        else if (d_well_count == acq_parameters.max_dwells)
        {
            d_state = 0;
            d_active = false;
            d_step_two = false;
            send_negative_acquisition();
        }
    }
    else
    {
        d_active = false;
        if (d_test_statistics > d_threshold)
        {
            if (acq_parameters.make_2_steps)
            {
                if (d_step_two)
                {
                    send_positive_acquisition();
                    d_step_two = false;
                    d_state = 0;  // Positive acquisition
                }
                else
                {
                    d_step_two = true;  // Clear input buffer and make small grid acquisition
                    d_state = 0;
                }
            }
            else
            {
                send_positive_acquisition();
                d_state = 0;  // Positive acquisition
            }
        }
        else
        {
            d_state = 0;  // Negative acquisition
            d_step_two = false;
            send_negative_acquisition();
        }
    }
    d_worker_active = false;
}

void pcps_cuda_acquisition::cuda_max_magt_sq_and_index(float* magt, int* magt_idx, cuComplex *in, unsigned int size) {
    gpuErrchk(cudaSetDevice(selected_device));
    int threads = size / cu_magt_blocks;
    cuda_max_magt_sq_and_index_stage1<<<cu_magt_blocks, threads, threads * (sizeof(float) + sizeof(int))>>>(cu_magt_tmp, cu_magt_idx_tmp, in, size);
    gpuErrchk(cudaGetLastError());
    cuda_max_magt_sq_and_index_stage2<<<1, cu_magt_blocks>>>(cu_magt_tmp, cu_magt_idx_tmp, size);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(magt, cu_magt_tmp, sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(magt_idx, cu_magt_idx_tmp, sizeof(int), cudaMemcpyDeviceToHost));
}

void pcps_cuda_acquisition::cuda_magt_sq_sum(float* sum, cuComplex *in, unsigned int size) {
    gpuErrchk(cudaSetDevice(selected_device));
    int threads = size / cu_magt_blocks;
    cuda_magt_sq_sum_stage1<<<cu_magt_blocks, threads, threads * sizeof(float)>>>(cu_magt_tmp, in, size);
    gpuErrchk(cudaGetLastError());
    cuda_magt_sq_sum_stage2<<<1, cu_magt_blocks>>>(cu_magt_tmp, size);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(sum, cu_magt_tmp, sizeof(float), cudaMemcpyDeviceToHost));
}

//bool ceq(gr_complex c1, gr_complex c2) {
//    return (fabs(c1.real() - c2.real()) < 2 && fabs(c1.imag() - c2.imag()) < 2);
//}
//
//void check_buff(gr_complex* cpu, cuComplex* gpu, unsigned int size, const char* error) {
//    gr_complex* gcpu = (gr_complex*)malloc(sizeof(gr_complex) * size);
//    gpuErrchk(cudaMemcpy(gcpu, gpu, sizeof(gr_complex) * size, cudaMemcpyDeviceToHost));
//    for(int i = 0; i < size; i++) {
//        if(!ceq(cpu[i],gcpu[i])) {
//            std::cout << "elem: " << i << ", expected: " << cpu[i] << ", found: " << gcpu[i] << " " << error << std::endl;
//            return;
//        }
//    }
//    free(gcpu);
//}

//void pcps_cuda_acquisition::acquisition_core_cuda() {
//    int doppler;
//    uint32_t indext = 0;
//    float magt = 0.0;
//    float fft_normalization_factor = static_cast<float>(d_fft_size) * static_cast<float>(d_fft_size);
//    gr_complex *in = d_in_buffer[d_well_count];
//    unsigned long int samplestamp = d_sample_counter_buffer[d_well_count];
//
//    d_input_power = 0.0;
//    d_mag = 0.0;
//
//    gpuErrchk(cudaMemcpy(cu_buffer_in, in, sizeof(cuComplex) * d_fft_size, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpy(cu_buffer_in + d_fft_size, d_zero_vector, sizeof(cuComplex) * (d_fft_size_pow2 - d_fft_size), cudaMemcpyHostToDevice));
//
//    d_well_count++;
//
//    DLOG(INFO) << "Channel: " << d_channel
//               << " , doing acquisition of satellite: " << d_gnss_synchro->System << " " << d_gnss_synchro->PRN
//               << " ,sample stamp: " << d_sample_counter << ", threshold: "
//               << d_threshold << ", doppler_max: " << d_doppler_max
//               << ", doppler_step: " << d_doppler_step;
//
//    // 1- Compute the input signal power estimation
//    volk_32fc_magnitude_squared_32f(d_magnitude, in, d_fft_size);
//    volk_32f_accumulator_s32f(&d_input_power, d_magnitude, d_fft_size);
//    d_input_power /= static_cast<float>(d_fft_size);
//    gr_complex* test_buff = (gr_complex*)malloc(sizeof(gr_complex) * d_fft_size_pow2);
//
//    for (unsigned int doppler_index = 0; doppler_index < d_num_doppler_bins; doppler_index++) {
//        // doppler search steps
//        doppler = -static_cast<int>(d_doppler_max) + d_doppler_step * doppler_index;
//
//        cuda_mul_vectors<<<cu_num_blocks, cu_num_threads>>>(cu_fft_buffer, cu_buffer_in,
//                cu_buffer_grid_doppler_wipeoffs[doppler_index], d_fft_size_pow2);
//        gpuErrchk(cudaGetLastError());
//
//        cufftErrchk(cufftExecC2C(cu_fft_plan, cu_fft_buffer, cu_ifft_buffer, CUFFT_FORWARD));
//
//        cuda_mul_vectors<<<cu_num_blocks, cu_num_threads>>>(cu_ifft_buffer, cu_ifft_buffer,
//                cu_buffer_fft_codes, d_fft_size_pow2);
//        gpuErrchk(cudaGetLastError());
//
//        cufftErrchk(cufftExecC2C(cu_fft_plan, cu_ifft_buffer, cu_ifft_buffer, CUFFT_INVERSE));
//
//        cuda_max_magt_sq_and_index(&magt, (int *)&indext, cu_ifft_buffer, d_fft_size_pow2);
//        magt /= (fft_normalization_factor * fft_normalization_factor);
//
//        // 4- record the maximum peak and the associated synchronization parameters
//        if (d_mag < magt)
//        {
//            d_mag = magt;
//
//            // In case that d_bit_transition_flag = true, we compare the potentially
//            // new maximum test statistics (d_mag/d_input_power) with the value in
//            // d_test_statistics. When the second dwell is being processed, the value
//            // of d_mag/d_input_power could be lower than d_test_statistics (i.e,
//            // the maximum test statistics in the previous dwell is greater than
//            // current d_mag/d_input_power). Note that d_test_statistics is not
//            // restarted between consecutive dwells in multidwell operation.
//            if (d_test_statistics < (d_mag / d_input_power) || !d_bit_transition_flag)
//            {
//                d_gnss_synchro->Acq_delay_samples = static_cast<double>(indext % d_samples_per_code);
//                d_gnss_synchro->Acq_doppler_hz = static_cast<double>(doppler);
//                d_gnss_synchro->Acq_samplestamp_samples = samplestamp;
//
//                // 5- Compute the test statistics and compare to the threshold
//                //d_test_statistics = 2 * d_fft_size * d_mag / d_input_power;
//                d_test_statistics = d_mag / d_input_power;
//            }
//        }
//
//        // Record results to file if required
//        if (d_dump)
//        {
//            std::stringstream filename;
//            std::streamsize n = 2 * sizeof(float) * (d_fft_size);  // complex file write
//            filename.str("");
//            filename << "../data/test_statistics_" << d_gnss_synchro->System
//                     << "_" << d_gnss_synchro->Signal << "_sat_"
//                     << d_gnss_synchro->PRN << "_doppler_" << doppler << ".dat";
//            d_dump_file.open(filename.str().c_str(), std::ios::out | std::ios::binary);
//            d_dump_file.write(reinterpret_cast<char *>(d_ifft->get_outbuf()), n);  //write directly |abs(x)|^2 in this Doppler bin?
//            d_dump_file.close();
//        }
//    }
//
//    //    gettimeofday(&tv, NULL);
//    //    end = tv.tv_sec *1e6 + tv.tv_usec;
//    //    std::cout << "Acq time = " << (end-begin) << " us" << std::endl;
//
//    if (!d_bit_transition_flag)
//    {
//        if (d_test_statistics > d_threshold)
//        {
//            d_state = 2;  // Positive acquisition
//        }
//        else if (d_well_count == d_max_dwells)
//        {
//            d_state = 3;  // Negative acquisition
//        }
//    }
//    else
//    {
//        if (d_well_count == d_max_dwells)  // d_max_dwells = 2
//        {
//            if (d_test_statistics > d_threshold)
//            {
//                d_state = 2;  // Positive acquisition
//            }
//            else
//            {
//                d_state = 3;  // Negative acquisition
//            }
//        }
//    }
//
//    d_core_working = false;
//}

int pcps_cuda_acquisition::general_work(int noutput_items __attribute__((unused)),
                                   gr_vector_int& ninput_items, gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items __attribute__((unused)))
{
    /*
     * By J.Arribas, L.Esteve and M.Molina
     * Acquisition strategy (Kay Borre book + CFAR threshold):
     * 1. Compute the input signal power estimation
     * 2. Doppler serial search loop
     * 3. Perform the FFT-based circular convolution (parallel time search)
     * 4. Record the maximum peak and the associated synchronization parameters
     * 5. Compute the test statistics and compare to the threshold
     * 6. Declare positive or negative acquisition using a message port
     */
    gpuErrchk(cudaSetDevice(selected_device));
    gr::thread::scoped_lock lk(d_setlock);
    if (!d_active or d_worker_active)
    {
        d_sample_counter += d_fft_size * ninput_items[0];
        consume_each(ninput_items[0]);
        if (d_step_two)
        {
            d_doppler_center_step_two = static_cast<float>(d_gnss_synchro->Acq_doppler_hz);
            update_grid_doppler_wipeoffs_step2();
            d_state = 0;
            d_active = true;
        }
        return 0;
    }

    switch (d_state)
    {
        case 0:
        {
            //restart acquisition variables
            d_gnss_synchro->Acq_delay_samples = 0.0;
            d_gnss_synchro->Acq_doppler_hz = 0.0;
            d_gnss_synchro->Acq_samplestamp_samples = 0;
            d_well_count = 0;
            d_mag = 0.0;
            d_input_power = 0.0;
            d_test_statistics = 0.0;
            d_state = 1;
            d_sample_counter += d_fft_size * ninput_items[0];  // sample counter
            consume_each(ninput_items[0]);
            break;
        }

        case 1:
        {
            // Copy the data to the core and let it know that new data is available
            if (d_cshort)
            {
                gpuErrchk(cudaMemcpy(cu_data_buffer_sc, input_items[0], d_fft_size * sizeof(lv_16sc_t), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemset(cu_data_buffer_sc + d_fft_size, 0, d_fft_padding_len * sizeof(lv_16sc_t)));
            }
            else
            {
                gpuErrchk(cudaMemcpy(cu_data_buffer, input_items[0], d_fft_size * sizeof(cuComplex), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemset(cu_data_buffer + d_fft_size, 0, d_fft_padding_len * sizeof(cuComplex)));

            }
            if (acq_parameters.blocking)
            {
                lk.unlock();
                acquisition_core(d_sample_counter);
            }
            else
            {
                gr::thread::thread d_worker(&pcps_cuda_acquisition::acquisition_core, this, d_sample_counter);
                d_worker_active = true;
            }
            d_sample_counter += d_fft_size;
            consume_each(1);
            break;
        }
    }
    return 0;
}
