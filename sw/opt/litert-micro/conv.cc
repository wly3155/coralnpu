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

#define CONV_MAC(in_ptr, fil)                                \
  asm("vsetvli zero, %[vl], e16, m2, ta, ma;"                \
      "vle8.v v28, %[input_ptr];"                            \
      "vsext.vf2 v26, v28;"                                  \
      "vadd.vx v26, v26, %[input_offset];"                   \
      "vsext.vf2 v28, %[filter];"                            \
      "vwmacc.vv %[acc], v26, v28;"                          \
      : [acc] "+vr"(mul_acc)                                 \
      : [vl] "r"(vl), [input_ptr] "A"(*in_ptr),              \
        [input_offset] "r"(input_offset), [filter] "vr"(fil) \
      : "v26", "v27", "v28", "v29", "vl", "vtype");

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

void Conv_4_4_16(const ConvParams& params, const OpDataConvCustom& data,
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

          vint32m4_t mul_acc;
          mul_acc = __riscv_vmv_v_x_i32m4(0, 16);

          const int8_t* in_ptrs[4][4];
          for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
              in_ptrs[r][c] = base_ptr + r * row_stride + c * col_stride;
            }
          }
          if (in_y_origin >= 0 && in_y_origin + 3 < input_height &&
              in_x_origin >= 0 && in_x_origin + 3 < input_width) {
            // Fast Path: Entirely inside the image
            CONV_MAC(in_ptrs[0][0], fil00);
            CONV_MAC(in_ptrs[0][1], fil01);
            CONV_MAC(in_ptrs[0][2], fil02);
            CONV_MAC(in_ptrs[0][3], fil03);
            CONV_MAC(in_ptrs[1][0], fil10);
            CONV_MAC(in_ptrs[1][1], fil11);
            CONV_MAC(in_ptrs[1][2], fil12);
            CONV_MAC(in_ptrs[1][3], fil13);
            CONV_MAC(in_ptrs[2][0], fil20);
            CONV_MAC(in_ptrs[2][1], fil21);
            CONV_MAC(in_ptrs[2][2], fil22);
            CONV_MAC(in_ptrs[2][3], fil23);
            CONV_MAC(in_ptrs[3][0], fil30);
            CONV_MAC(in_ptrs[3][1], fil31);
            CONV_MAC(in_ptrs[3][2], fil32);
            CONV_MAC(in_ptrs[3][3], fil33);
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
                CONV_MAC(in_ptrs[0][0], fil00);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[0][1], fil01);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[0][2], fil02);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[0][3], fil03);
              }
            }
            if (rv1) {
              if (cv0) {
                CONV_MAC(in_ptrs[1][0], fil10);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[1][1], fil11);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[1][2], fil12);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[1][3], fil13);
              }
            }
            if (rv2) {
              if (cv0) {
                CONV_MAC(in_ptrs[2][0], fil20);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[2][1], fil21);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[2][2], fil22);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[2][3], fil23);
              }
            }
            if (rv3) {
              if (cv0) {
                CONV_MAC(in_ptrs[3][0], fil30);
              }
              if (cv1) {
                CONV_MAC(in_ptrs[3][1], fil31);
              }
              if (cv2) {
                CONV_MAC(in_ptrs[3][2], fil32);
              }
              if (cv3) {
                CONV_MAC(in_ptrs[3][3], fil33);
              }
            }
          }
          int32_t temp_acc =
              __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(
                  mul_acc, __riscv_vmv_v_x_i32m1(0, 1), vl));
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
