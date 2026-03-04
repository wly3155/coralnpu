// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sw/opt/litert-micro/conv.h"

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "sw/opt/litert-micro/accumulator_util.h"
#include "sw/opt/litert-micro/memory_util.h"
#include "sw/opt/rvv_opt.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifdef USE_TFLM_COMPRESSION
#error "USE_TFLM_COMPRESSION is not supported"
#endif  // USE_TFLM_COMPRESSION

// Leverage compiler register allocator, but inline assembly MAC

#define CONV_MAC(in_ptr, fil, acc_reg)                       \
  asm("vsetvli zero, %[vl], e16, m2, ta, ma;"                \
      "vle8.v v30, %[input_ptr];"                            \
      "vsext.vf2 v18, v30;"                                  \
      "vadd.vx v18, v18, %[input_offset];"                   \
      "vsext.vf2 v30, %[filter];"                            \
      "vwmacc.vv %[acc], v18, v30;"                          \
      : [acc] "+vr"(acc_reg)                                 \
      : [vl] "r"(vl), [input_ptr] "A"(*in_ptr),              \
        [input_offset] "r"(input_offset), [filter] "vr"(fil) \
      : "v18", "v19", "v30", "v31", "vl", "vtype");

#define CONV_MAC_2X(in_ptr, fil_for_acc1, fil_for_acc2)              \
  asm("vsetvli zero, %[vl], e16, m2, ta, ma;"                        \
      "vle8.v v30, %[input_ptr];"                                    \
      "vsext.vf2 v18, v30;"                                          \
      "vadd.vx v18, v18, %[input_offset];"                           \
      "vsext.vf2 v30, %[fil1];"                                      \
      "vwmacc.vv %[acc1], v18, v30;"                                 \
      "vsext.vf2 v30, %[fil2];"                                      \
      "vwmacc.vv %[acc2], v18, v30;"                                 \
      : [acc1] "+vr"(mul_acc1), [acc2] "+vr"(mul_acc2)               \
      : [vl] "r"(vl), [input_ptr] "A"(*in_ptr),                      \
        [input_offset] "r"(input_offset), [fil1] "vr"(fil_for_acc1), \
        [fil2] "vr"(fil_for_acc2)                                    \
      : "v18", "v19", "v30", "v31", "vl", "vtype");

namespace coralnpu_v2::opt::litert_micro {

using tflite::ConvParams;
using tflite::kConvBiasTensor;
using tflite::kConvInputTensor;
using tflite::kConvOutputTensor;
using tflite::kConvWeightsTensor;
using tflite::NumInputs;
using tflite::OpDataConv;
using tflite::RuntimeShape;
using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetOptionalTensorData;
using tflite::micro::GetTensorData;
using tflite::micro::GetTensorShape;

void Conv_4_4_16_StrideN(
    const ConvParams& params, const OpDataConvCustom& data,
    const int32_t* output_multiplier, const uint8_t* shift_left,
    const uint8_t* shift_right, TfLiteContext* context,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int16_t input_offset = params.input_offset;  // r = s(q - Z)
  const auto output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const auto stride_width = params.stride_width;
  const auto stride_height = params.stride_height;
  const auto pad_width = params.padding_values.width;
  const auto pad_height = params.padding_values.height;
  const auto input_height = input_shape.Dims(1);
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);

  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  TFLITE_DCHECK_EQ(filter_height, 4);
  TFLITE_DCHECK_EQ(filter_width, 4);
  TFLITE_DCHECK_LE(input_depth, 16);

  const auto output_height = output_shape.Dims(1);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  size_t vl = __riscv_vsetvl_e8m1(input_depth);
  const int row_stride = input_width * input_depth;
  const int col_stride = input_depth;
  const int row_step = stride_height * row_stride;
  const int col_step = stride_width * col_stride;

  const int filter_row_stride = filter_shape.Dims(2) * input_depth;
  const int filter_col_stride = input_depth;

  int32_t* accs_buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.accs_buffer_index));
  TFLITE_DCHECK_NE(accs_buf, nullptr);
  // Clear the accumulator buffer
  Memset(
      accs_buf, 0,
      batches * output_height * output_width * output_depth * sizeof(int32_t));

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    const int8_t* filter_base_ptr =
        &filter_data[Offset(filter_shape, out_channel, 0, 0, 0)];
    register vint8m1_t fil00 __asm__("v1");
    register vint8m1_t fil01 __asm__("v2");
    register vint8m1_t fil02 __asm__("v3");
    register vint8m1_t fil03 __asm__("v4");
    register vint8m1_t fil10 __asm__("v5");
    register vint8m1_t fil11 __asm__("v6");
    register vint8m1_t fil12 __asm__("v7");
    register vint8m1_t fil13 __asm__("v8");
    register vint8m1_t fil20 __asm__("v9");
    register vint8m1_t fil21 __asm__("v10");
    register vint8m1_t fil22 __asm__("v11");
    register vint8m1_t fil23 __asm__("v12");
    register vint8m1_t fil30 __asm__("v13");
    register vint8m1_t fil31 __asm__("v14");
    register vint8m1_t fil32 __asm__("v15");
    register vint8m1_t fil33 __asm__("v16");

    fil00 = __riscv_vle8_v_i8m1(filter_base_ptr, vl);
    fil01 = __riscv_vle8_v_i8m1(filter_base_ptr + 1 * filter_col_stride, vl);
    fil02 = __riscv_vle8_v_i8m1(filter_base_ptr + 2 * filter_col_stride, vl);
    fil03 = __riscv_vle8_v_i8m1(filter_base_ptr + 3 * filter_col_stride, vl);
    fil10 = __riscv_vle8_v_i8m1(filter_base_ptr + filter_row_stride, vl);
    fil11 = __riscv_vle8_v_i8m1(
        filter_base_ptr + filter_row_stride + 1 * filter_col_stride, vl);
    fil12 = __riscv_vle8_v_i8m1(
        filter_base_ptr + filter_row_stride + 2 * filter_col_stride, vl);
    fil13 = __riscv_vle8_v_i8m1(
        filter_base_ptr + filter_row_stride + 3 * filter_col_stride, vl);
    fil20 = __riscv_vle8_v_i8m1(filter_base_ptr + 2 * filter_row_stride, vl);
    fil21 = __riscv_vle8_v_i8m1(
        filter_base_ptr + 2 * filter_row_stride + 1 * filter_col_stride, vl);
    fil22 = __riscv_vle8_v_i8m1(
        filter_base_ptr + 2 * filter_row_stride + 2 * filter_col_stride, vl);
    fil23 = __riscv_vle8_v_i8m1(
        filter_base_ptr + 2 * filter_row_stride + 3 * filter_col_stride, vl);
    fil30 = __riscv_vle8_v_i8m1(filter_base_ptr + 3 * filter_row_stride, vl);
    fil31 = __riscv_vle8_v_i8m1(
        filter_base_ptr + 3 * filter_row_stride + 1 * filter_col_stride, vl);
    fil32 = __riscv_vle8_v_i8m1(
        filter_base_ptr + 3 * filter_row_stride + 2 * filter_col_stride, vl);
    fil33 = __riscv_vle8_v_i8m1(
        filter_base_ptr + 3 * filter_row_stride + 3 * filter_col_stride, vl);

    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* batch_base_ptr =
          &input_data[Offset(input_shape, batch, 0, 0, 0)];
      const int8_t* row_ptr =
          batch_base_ptr - pad_height * row_stride - pad_width * col_stride;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int8_t* base_ptr = row_ptr;

        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;

          vint32m4_t mul_acc1;
          mul_acc1 = __riscv_vmv_v_x_i32m4(0, 16);

          const int8_t* in_ptrs[4][4];
          for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
              in_ptrs[r][c] = base_ptr + r * row_stride + c * col_stride;
            }
          }
          if (in_y_origin >= 0 && in_y_origin + 3 < input_height &&
              in_x_origin >= 0 && in_x_origin + 3 < input_width) {
            // Fast Path: Entirely inside the image
            CONV_MAC(in_ptrs[0][0], fil00, mul_acc1);
            CONV_MAC(in_ptrs[0][1], fil01, mul_acc1);
            CONV_MAC(in_ptrs[0][2], fil02, mul_acc1);
            CONV_MAC(in_ptrs[0][3], fil03, mul_acc1);
            CONV_MAC(in_ptrs[1][0], fil10, mul_acc1);
            CONV_MAC(in_ptrs[1][1], fil11, mul_acc1);
            CONV_MAC(in_ptrs[1][2], fil12, mul_acc1);
            CONV_MAC(in_ptrs[1][3], fil13, mul_acc1);
            CONV_MAC(in_ptrs[2][0], fil20, mul_acc1);
            CONV_MAC(in_ptrs[2][1], fil21, mul_acc1);
            CONV_MAC(in_ptrs[2][2], fil22, mul_acc1);
            CONV_MAC(in_ptrs[2][3], fil23, mul_acc1);
            CONV_MAC(in_ptrs[3][0], fil30, mul_acc1);
            CONV_MAC(in_ptrs[3][1], fil31, mul_acc1);
            CONV_MAC(in_ptrs[3][2], fil32, mul_acc1);
            CONV_MAC(in_ptrs[3][3], fil33, mul_acc1);
          } else {
            // Slow Path: Crosses boundaries, handle with guards
            const bool rv0 =
                (in_y_origin + 0 >= 0) && (in_y_origin + 0 < input_height);
            const bool rv1 =
                (in_y_origin + 1 >= 0) && (in_y_origin + 1 < input_height);
            const bool rv2 =
                (in_y_origin + 2 >= 0) && (in_y_origin + 2 < input_height);
            const bool rv3 =
                (in_y_origin + 3 >= 0) && (in_y_origin + 3 < input_height);

            const bool cv0 =
                (in_x_origin + 0 >= 0) && (in_x_origin + 0 < input_width);
            const bool cv1 =
                (in_x_origin + 1 >= 0) && (in_x_origin + 1 < input_width);
            const bool cv2 =
                (in_x_origin + 2 >= 0) && (in_x_origin + 2 < input_width);
            const bool cv3 =
                (in_x_origin + 3 >= 0) && (in_x_origin + 3 < input_width);

            if (rv0) {
              if (cv0) {
                CONV_MAC(in_ptrs[0][0], fil00, mul_acc1);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[0][1], fil01, mul_acc1);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[0][2], fil02, mul_acc1);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[0][3], fil03, mul_acc1);
              }
            }
            if (rv1) {
              if (cv0) {
                CONV_MAC(in_ptrs[1][0], fil10, mul_acc1);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[1][1], fil11, mul_acc1);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[1][2], fil12, mul_acc1);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[1][3], fil13, mul_acc1);
              }
            }
            if (rv2) {
              if (cv0) {
                CONV_MAC(in_ptrs[2][0], fil20, mul_acc1);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[2][1], fil21, mul_acc1);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[2][2], fil22, mul_acc1);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[2][3], fil23, mul_acc1);
              }
            }
            if (rv3) {
              if (cv0) {
                CONV_MAC(in_ptrs[3][0], fil30, mul_acc1);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[3][1], fil31, mul_acc1);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[3][2], fil32, mul_acc1);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[3][3], fil33, mul_acc1);
              }
            }
          }
          int32_t temp_acc =
              __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(
                  mul_acc1, __riscv_vmv_v_x_i32m1(0, 1), vl));
          accs_buf[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              temp_acc;
          base_ptr += col_step;
        }
        row_ptr += row_step;
      }
    }
  }

  // Post process the entire batch of accumulators at once
  PostprocessAcc(accs_buf, bias_data, shift_left, output_multiplier,
                 shift_right, output_offset, output_activation_min,
                 output_activation_max, output_data,
                 batches * output_height * output_width, output_depth);
}

// Kernel for 4x4 filter, 48 input channels, stride 1
// Strategy:
// - Divide 48 input channels into 3 chunks of 16.
// - Outer loops: batch, output_channel.
// - Inner loop: chunk (0..2).
//   - Load 4x4x16 filter path into registers v1-v16.
//   - Loop over spatial dimensions (out_y, out_x).
//     - Compute 2x output pixels (out_x, out_x+1).
//     - Accumulate into accs_buf.
void Conv_4_4_48_Stride1(
    const ConvParams& params, const OpDataConvCustom& data,
    const int32_t* output_multiplier, const uint8_t* shift_left,
    const uint8_t* shift_right, TfLiteContext* context,
    const RuntimeShape& input_shape, const int8_t* input_data,
    const RuntimeShape& filter_shape, const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int filter_row_stride = filter_shape.Dims(2) * input_depth;
  const int filter_col_stride = input_depth;
  const int32_t input_offset = params.input_offset;

  int32_t* accs_buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.accs_buffer_index));
  TFLITE_DCHECK_NE(accs_buf, nullptr);
  // Clear the accumulator buffer
  Memset(
      accs_buf, 0,
      batches * output_height * output_width * output_depth * sizeof(int32_t));

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
      const int8_t* filter_base_ptr =
          filter_data + Offset(filter_shape, out_channel, 0, 0, 0);

      int rem_channels = input_depth;
      // Divide input depth (48) into 3 chunks of 16 (or more for general depth)
      // Use (input_depth + 15) / 16 to determine number of chunks
      int num_chunks = (input_depth + 15) / 16;
      for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int8_t* chunk_ptr = filter_base_ptr + chunk * 16;

        // Pin filter patch to registers to match Conv_4_4_16 strategy
        register vint8m1_t fil00 __asm__("v1");
        register vint8m1_t fil01 __asm__("v2");
        register vint8m1_t fil02 __asm__("v3");
        register vint8m1_t fil03 __asm__("v4");
        register vint8m1_t fil10 __asm__("v5");
        register vint8m1_t fil11 __asm__("v6");
        register vint8m1_t fil12 __asm__("v7");
        register vint8m1_t fil13 __asm__("v8");
        register vint8m1_t fil20 __asm__("v9");
        register vint8m1_t fil21 __asm__("v10");
        register vint8m1_t fil22 __asm__("v11");
        register vint8m1_t fil23 __asm__("v12");
        register vint8m1_t fil30 __asm__("v13");
        register vint8m1_t fil31 __asm__("v14");
        register vint8m1_t fil32 __asm__("v15");
        register vint8m1_t fil33 __asm__("v16");

        size_t vl = __riscv_vsetvl_e8m1(rem_channels);
        rem_channels -= 16;

        fil00 = __riscv_vle8_v_i8m1(chunk_ptr, vl);
        fil01 = __riscv_vle8_v_i8m1(chunk_ptr + 1 * filter_col_stride, vl);
        fil02 = __riscv_vle8_v_i8m1(chunk_ptr + 2 * filter_col_stride, vl);
        fil03 = __riscv_vle8_v_i8m1(chunk_ptr + 3 * filter_col_stride, vl);
        fil10 = __riscv_vle8_v_i8m1(chunk_ptr + filter_row_stride, vl);
        fil11 = __riscv_vle8_v_i8m1(
            chunk_ptr + filter_row_stride + 1 * filter_col_stride, vl);
        fil12 = __riscv_vle8_v_i8m1(
            chunk_ptr + filter_row_stride + 2 * filter_col_stride, vl);
        fil13 = __riscv_vle8_v_i8m1(
            chunk_ptr + filter_row_stride + 3 * filter_col_stride, vl);
        fil20 = __riscv_vle8_v_i8m1(chunk_ptr + 2 * filter_row_stride, vl);
        fil21 = __riscv_vle8_v_i8m1(
            chunk_ptr + 2 * filter_row_stride + 1 * filter_col_stride, vl);
        fil22 = __riscv_vle8_v_i8m1(
            chunk_ptr + 2 * filter_row_stride + 2 * filter_col_stride, vl);
        fil23 = __riscv_vle8_v_i8m1(
            chunk_ptr + 2 * filter_row_stride + 3 * filter_col_stride, vl);
        fil30 = __riscv_vle8_v_i8m1(chunk_ptr + 3 * filter_row_stride, vl);
        fil31 = __riscv_vle8_v_i8m1(
            chunk_ptr + 3 * filter_row_stride + 1 * filter_col_stride, vl);
        fil32 = __riscv_vle8_v_i8m1(
            chunk_ptr + 3 * filter_row_stride + 2 * filter_col_stride, vl);
        fil33 = __riscv_vle8_v_i8m1(
            chunk_ptr + 3 * filter_row_stride + 3 * filter_col_stride, vl);

        const int row_stride = input_width * input_depth;
        const int col_stride = input_depth;

        const int pad_width = params.padding_values.width;
        const int pad_height = params.padding_values.height;

        const int8_t* base_ptr =
            input_data + Offset(input_shape, batch, 0, 0, chunk * 16);

        // Loop over spatial dimensions
        for (int out_y = 0; out_y < output_height; ++out_y) {
          const int in_y_origin = out_y - pad_height;
          const int8_t* row_ptr = base_ptr + in_y_origin * row_stride;

          for (int out_x = 0; out_x < output_width; out_x += 2) {
            const int in_x_origin1 = out_x - pad_width;
            const int in_x_origin2 = out_x + 1 - pad_width;

            // Pointers to the 16-element chunks
            const int8_t* in_ptrs[4][5];
            // Initialize assuming fast path first
            const int8_t* curr_ptr = row_ptr + in_x_origin1 * col_stride;

            for (int r = 0; r < 4; ++r) {
              for (int c = 0; c < 5; ++c) {
                in_ptrs[r][c] = curr_ptr + r * row_stride + c * col_stride;
              }
            }

            // Accumulators
            register vint32m4_t mul_acc1 __asm__("v20");
            register vint32m4_t mul_acc2 __asm__("v24");
            mul_acc1 = __riscv_vmv_v_x_i32m4(0, 16);
            mul_acc2 = __riscv_vmv_v_x_i32m4(0, 16);

            // If chunk 0: Initialize with 0 (bias added later in PostProcess?
            // Actually conv kernels usually add bias at end.
            // We can just zero-init here.
            // ... Perform Convolution similar to Stride1 ...
            if (in_y_origin >= 0 && in_y_origin + 3 < input_height &&
                in_x_origin1 >= 0 && in_x_origin2 + 3 < input_width) {
              // Fast Path
              CONV_MAC(in_ptrs[0][0], fil00, mul_acc1);
              CONV_MAC_2X(in_ptrs[0][1], fil01, fil00);
              CONV_MAC_2X(in_ptrs[0][2], fil02, fil01);
              CONV_MAC_2X(in_ptrs[0][3], fil03, fil02);
              CONV_MAC(in_ptrs[0][4], fil03, mul_acc2);

              CONV_MAC(in_ptrs[1][0], fil10, mul_acc1);
              CONV_MAC_2X(in_ptrs[1][1], fil11, fil10);
              CONV_MAC_2X(in_ptrs[1][2], fil12, fil11);
              CONV_MAC_2X(in_ptrs[1][3], fil13, fil12);
              CONV_MAC(in_ptrs[1][4], fil13, mul_acc2);

              CONV_MAC(in_ptrs[2][0], fil20, mul_acc1);
              CONV_MAC_2X(in_ptrs[2][1], fil21, fil20);
              CONV_MAC_2X(in_ptrs[2][2], fil22, fil21);
              CONV_MAC_2X(in_ptrs[2][3], fil23, fil22);
              CONV_MAC(in_ptrs[2][4], fil23, mul_acc2);

              CONV_MAC(in_ptrs[3][0], fil30, mul_acc1);
              CONV_MAC_2X(in_ptrs[3][1], fil31, fil30);
              CONV_MAC_2X(in_ptrs[3][2], fil32, fil31);
              CONV_MAC_2X(in_ptrs[3][3], fil33, fil32);
              CONV_MAC(in_ptrs[3][4], fil33, mul_acc2);
            } else {
              // Slow path logic (omitted for brevity, can copy from Stride1 if
              // needed or assume padding handled?) For now, let's copy the slow
              // path logic but adapted Copy-paste slow path from Stride1...
              const bool rv[4] = {
                  (in_y_origin + 0 >= 0) && (in_y_origin + 0 < input_height),
                  (in_y_origin + 1 >= 0) && (in_y_origin + 1 < input_height),
                  (in_y_origin + 2 >= 0) && (in_y_origin + 2 < input_height),
                  (in_y_origin + 3 >= 0) && (in_y_origin + 3 < input_height)};
              const bool cv1[4] = {
                  (in_x_origin1 + 0 >= 0) && (in_x_origin1 + 0 < input_width),
                  (in_x_origin1 + 1 >= 0) && (in_x_origin1 + 1 < input_width),
                  (in_x_origin1 + 2 >= 0) && (in_x_origin1 + 2 < input_width),
                  (in_x_origin1 + 3 >= 0) && (in_x_origin1 + 3 < input_width)};
              // Reconstruct slow path input pointers to handle OOB
              const int8_t* in_ptrs1[4][4];
              const int8_t* in_ptrs2[4][4];
              for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                  in_ptrs1[r][c] = curr_ptr + r * row_stride + c * col_stride;
                  in_ptrs2[r][c] = in_ptrs1[r][c] + col_stride;
                }
              }

              if (rv[0]) {
                if (cv1[0]) CONV_MAC(in_ptrs1[0][0], fil00, mul_acc1);
                if (cv1[1]) CONV_MAC(in_ptrs1[0][1], fil01, mul_acc1);
                if (cv1[2]) CONV_MAC(in_ptrs1[0][2], fil02, mul_acc1);
                if (cv1[3]) CONV_MAC(in_ptrs1[0][3], fil03, mul_acc1);
              }
              if (rv[1]) {
                if (cv1[0]) CONV_MAC(in_ptrs1[1][0], fil10, mul_acc1);
                if (cv1[1]) CONV_MAC(in_ptrs1[1][1], fil11, mul_acc1);
                if (cv1[2]) CONV_MAC(in_ptrs1[1][2], fil12, mul_acc1);
                if (cv1[3]) CONV_MAC(in_ptrs1[1][3], fil13, mul_acc1);
              }
              if (rv[2]) {
                if (cv1[0]) CONV_MAC(in_ptrs1[2][0], fil20, mul_acc1);
                if (cv1[1]) CONV_MAC(in_ptrs1[2][1], fil21, mul_acc1);
                if (cv1[2]) CONV_MAC(in_ptrs1[2][2], fil22, mul_acc1);
                if (cv1[3]) CONV_MAC(in_ptrs1[2][3], fil23, mul_acc1);
              }
              if (rv[3]) {
                if (cv1[0]) CONV_MAC(in_ptrs1[3][0], fil30, mul_acc1);
                if (cv1[1]) CONV_MAC(in_ptrs1[3][1], fil31, mul_acc1);
                if (cv1[2]) CONV_MAC(in_ptrs1[3][2], fil32, mul_acc1);
                if (cv1[3]) CONV_MAC(in_ptrs1[3][3], fil33, mul_acc1);
              }

              const bool cv2[4] = {
                  (in_x_origin2 + 0 >= 0) && (in_x_origin2 + 0 < input_width),
                  (in_x_origin2 + 1 >= 0) && (in_x_origin2 + 1 < input_width),
                  (in_x_origin2 + 2 >= 0) && (in_x_origin2 + 2 < input_width),
                  (in_x_origin2 + 3 >= 0) && (in_x_origin2 + 3 < input_width)};

              if (rv[0]) {
                if (cv2[0]) CONV_MAC(in_ptrs2[0][0], fil00, mul_acc2);
                if (cv2[1]) CONV_MAC(in_ptrs2[0][1], fil01, mul_acc2);
                if (cv2[2]) CONV_MAC(in_ptrs2[0][2], fil02, mul_acc2);
                if (cv2[3]) CONV_MAC(in_ptrs2[0][3], fil03, mul_acc2);
              }
              if (rv[1]) {
                if (cv2[0]) CONV_MAC(in_ptrs2[1][0], fil10, mul_acc2);
                if (cv2[1]) CONV_MAC(in_ptrs2[1][1], fil11, mul_acc2);
                if (cv2[2]) CONV_MAC(in_ptrs2[1][2], fil12, mul_acc2);
                if (cv2[3]) CONV_MAC(in_ptrs2[1][3], fil13, mul_acc2);
              }
              if (rv[2]) {
                if (cv2[0]) CONV_MAC(in_ptrs2[2][0], fil20, mul_acc2);
                if (cv2[1]) CONV_MAC(in_ptrs2[2][1], fil21, mul_acc2);
                if (cv2[2]) CONV_MAC(in_ptrs2[2][2], fil22, mul_acc2);
                if (cv2[3]) CONV_MAC(in_ptrs2[2][3], fil23, mul_acc2);
              }
              if (rv[3]) {
                if (cv2[0]) CONV_MAC(in_ptrs2[3][0], fil30, mul_acc2);
                if (cv2[1]) CONV_MAC(in_ptrs2[3][1], fil31, mul_acc2);
                if (cv2[2]) CONV_MAC(in_ptrs2[3][2], fil32, mul_acc2);
                if (cv2[3]) CONV_MAC(in_ptrs2[3][3], fil33, mul_acc2);
              }
            }

            // Reduce and Accumulate to Buffer
            // Reduce mul_acc1
            int32_t acc1_val =
                __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(
                    mul_acc1, __riscv_vmv_v_x_i32m1(0, 1), vl));
            // Reduce mul_acc2
            int32_t acc2_val =
                __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(
                    mul_acc2, __riscv_vmv_v_x_i32m1(0, 1), vl));

            // Read-Modify-Write to Buffer
            int idx1 = Offset(output_shape, batch, out_y, out_x, out_channel);
            int idx2 =
                Offset(output_shape, batch, out_y, out_x + 1, out_channel);

            if (chunk == 0) {
              accs_buf[idx1] = acc1_val;
              if (out_x + 1 < output_width) accs_buf[idx2] = acc2_val;
            } else {
              accs_buf[idx1] += acc1_val;
              if (out_x + 1 < output_width) accs_buf[idx2] += acc2_val;
            }
          }
        }
      }
    }
  }
  // After all batches processed
  PostprocessAcc(accs_buf, bias_data, shift_left, output_multiplier,
                 shift_right, output_offset, output_activation_min,
                 output_activation_max, output_data,
                 batches * output_height * output_width, output_depth);
}

void Conv_4_4_16(const ConvParams& params, const OpDataConvCustom& data,
                 const int32_t* output_multiplier, const uint8_t* shift_left,
                 const uint8_t* shift_right, TfLiteContext* context,
                 const RuntimeShape& input_shape, const int8_t* input_data,
                 const RuntimeShape& filter_shape, const int8_t* filter_data,
                 const RuntimeShape& bias_shape, const int32_t* bias_data,
                 const RuntimeShape& output_shape, int8_t* output_data) {
  // Todo add a Stride specific strategy for Stride == 1 and 2
  Conv_4_4_16_StrideN(params, data, output_multiplier, shift_left, shift_right,
                      context, input_shape, input_data, filter_shape,
                      filter_data, bias_shape, bias_data, output_shape,
                      output_data);
}

#undef CONV_MAC

void ConvPerChannel(const ConvParams& params, const OpDataConvCustom& data,
                    const int32_t* output_multiplier,
                    const int32_t* output_shift, TfLiteContext* context,
                    const RuntimeShape& input_shape, const int8_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int32_t* bias_data,
                    const RuntimeShape& output_shape, int8_t* output_data) {
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);

  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);

  // Copy filter and bias to dtcm.
  auto filter_data_copy =
      make_aligned_array<int8_t>(16, filter_shape.FlatSize(), filter_data);
  // TODO(davidgao): if allocation fails, don't copy, use orig
  TFLITE_DCHECK_NE(filter_data_copy, nullptr);

  aligned_array<int32_t> bias_data_copy;
  if (bias_data) {
    bias_data_copy = make_aligned_array<int32_t>(16, output_depth, bias_data);
    // TODO(davidgao): if allocation fails, don't copy, use orig
    TFLITE_DCHECK_NE(bias_data_copy, nullptr);
  }

  // Shifting from quantization params for vectorization
  auto shift_left = make_aligned_array<uint8_t>(16, output_depth);
  TFLITE_DCHECK_NE(shift_left, nullptr);
  auto shift_right = make_aligned_array<uint8_t>(16, output_depth);
  TFLITE_DCHECK_NE(shift_right, nullptr);
  PrepareShiftParams(shift_left.get(), shift_right.get(), output_shift,
                     output_depth);

  if (filter_height == 4 && filter_width == 4 && input_depth <= 16) {
    Conv_4_4_16(params, data, output_multiplier, shift_left.get(),
                shift_right.get(), context, input_shape, input_data,
                filter_shape, filter_data_copy.get(), bias_shape,
                bias_data_copy.get(), output_shape, output_data);

  } else if (filter_height == 4 && filter_width == 4 && input_depth <= 48 &&
             params.stride_width == 1 && params.stride_height == 1) {
    Conv_4_4_48_Stride1(params, data, output_multiplier, shift_left.get(),
                        shift_right.get(), context, input_shape, input_data,
                        filter_shape, filter_data_copy.get(), bias_shape,
                        bias_data_copy.get(), output_shape, output_data);
  } else {
    tflite::reference_integer_ops::ConvPerChannel(
        params, output_multiplier, output_shift, input_shape, input_data,
        filter_shape, filter_data, bias_shape, bias_data, output_shape,
        output_data);
  }
}

TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& data = *(static_cast<const OpDataConvCustom*>(node->user_data));

  TfLiteEvalTensor* output = GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* input = GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3) ? GetEvalInput(context, node, kConvBiasTensor)
                             : nullptr;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt8: {
          ConvPerChannel(
              tflite::ConvParamsQuantized(params, data), data,
              data.per_channel_output_multiplier, data.per_channel_output_shift,
              context, GetTensorShape(input), GetTensorData<int8_t>(input),
              GetTensorShape(filter), GetTensorData<int8_t>(filter),
              GetTensorShape(bias), GetOptionalTensorData<int32_t>(bias),
              GetTensorShape(output), GetTensorData<int8_t>(output));
          break;
        }
        default:
          MicroPrintf("Filter type %s (%d) for input type %s not supported.",
                      TfLiteTypeGetName(filter->type), filter->type,
                      TfLiteTypeGetName(input->type));
          return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* ConvInit(TfLiteContext* context, const char* buffer, size_t length) {
  // Default tflite::ConvInit as a custom structure (OpDataConvCustom) is used
  // to store the scratch buffer index for our full-tensor accumulator buffering
  // strategy, so we cannot use the default tflite::ConvInit.
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataConvCustom));
}

TfLiteStatus ConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, tflite::ConvPrepare(context, node));

  // A custom Prepare to allocate the full-tensor accumulator buffer used for
  // vectorized post-processing, saving the index in our custom data.
  OpDataConvCustom* data = static_cast<OpDataConvCustom*>(node->user_data);
  tflite::MicroContext* micro_context = tflite::GetMicroContext(context);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const int batches = output->dims->data[0];
  const int output_height = output->dims->data[1];
  const int output_width = output->dims->data[2];
  const int output_depth = output->dims->data[3];

  size_t required_bytes =
      batches * output_height * output_width * output_depth * sizeof(int32_t);

  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, required_bytes, &data->accs_buffer_index));

  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TFLMRegistration Register_CONV_2D() {
  auto registration = tflite::Register_CONV_2D();
  registration.init = ConvInit;
  registration.prepare = ConvPrepare;
  registration.invoke = ConvEval;
  return registration;
}

}  // namespace coralnpu_v2::opt::litert_micro
