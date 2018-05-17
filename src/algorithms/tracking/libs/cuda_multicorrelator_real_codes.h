/*!
 * \file cuda_multicorrelator.h
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

#ifndef GNSS_SDR_CUDA_MULTICORRELATOR_REAL_CODES_H_
#define GNSS_SDR_CUDA_MULTICORRELATOR_REAL_CODES_H_


#include <cuComplex.h>
#include <gnuradio/gr_complex.h>
/*!
 * \brief Class that implements carrier wipe-off and correlators.
 */
class cuda_multicorrelator_real_codes
{
public:
    cuda_multicorrelator_real_codes();
    ~cuda_multicorrelator_real_codes();
    bool init(int device, int max_signal_length_samples, int code_length_chips, int n_correlators);
    bool set_local_code_and_taps(int code_length_chips, const float *local_code_in, float *shifts_chips, int n_correlations);
    bool set_input_output_vectors(gr_complex *corr_out, cuComplex *sig_in);
    void update_local_code(int correlator_length_samples, float rem_code_phase_chips, float code_phase_step_chips);
    bool Carrier_wipeoff_multicorrelator_resampler(float rem_carrier_phase_in_rad, float phase_step_rad, float rem_code_phase_chips, float code_phase_step_chips, int signal_length_samples);
    bool free();

private:
    // Allocate the device input vectors
    int cu_selected_device;

    cuComplex *cu_sig_in;
    float *cu_local_codes_resampled;
    float *cu_local_code_in;
    cuComplex *cu_corr_out;
    cuComplex *cu_phase;
    cuComplex *cu_red_tmp;
    float *cu_shifts_chips;

    const gr_complex *d_sig_in;
    gr_complex *d_corr_out;
    int d_code_length_chips;
    int d_n_correlators;
    int cu_num_blocks;
    int cu_num_threads;
};


#endif /* GNSS_SDR_CUDA_MULTICORRELATOR_REAL_CODES_H_ */
