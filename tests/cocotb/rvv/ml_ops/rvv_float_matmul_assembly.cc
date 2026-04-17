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

#include <riscv_vector.h>

#include <cstdint>

constexpr size_t kLhsRows = 16;
constexpr size_t kRhsCols = 16;
constexpr size_t kInner = 48;

float lhs_input[kLhsRows * kInner]
    __attribute__((section(".data"), used, retain))
    __attribute__((aligned(16)));
float rhs_input[kInner * kRhsCols]
    __attribute__((section(".data"), used, retain))
    __attribute__((aligned(16)));
float result_output[kLhsRows * kRhsCols]
    __attribute__((section(".data"), used, retain))
    __attribute__((aligned(16)));

void MatMulF(size_t lhs_rows, size_t inner, size_t rhs_cols, const float* lhs,
             const float* rhs, float* result) {
  const size_t vlenb = __riscv_vlenb();

  // Create zero register for vfredusum
  // (vmv.v.i v0, 0 sets all bits to 0, which is exactly +0.0f in IEEE 754)
  asm volatile("vsetvli zero, %0, e32, m4, ta, ma;"
               "vmv.v.i v0, 0;"
               :
               : "r"(vlenb));

  for (size_t r = 0; r < lhs_rows; r++) {
    const float* lhs_data = lhs + (r * inner);
    float* result_row = result + (r * rhs_cols);
    for (size_t c = 0; c < rhs_cols; c++) {
      const float* rhs_data = rhs + (c * inner);

      // Reset accumulators
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(vlenb));
      asm volatile("vmv.v.i v8, 0");

      // Inner dot product loop
      size_t k = 0;
      size_t vl = vlenb;
      while (k < inner) {
        if (inner - k < vl) {
          vl = inner - k;
        }
        asm volatile(
            "vsetvli zero, %[vl], e32, m4, ta, ma \n\t"
            "vle32.v  v12, (%[lhs_ptr]) \n\t"
            "vle32.v  v16, (%[rhs_ptr]) \n\t"
            "vfmul.vv v20, v12, v16 \n\t"
            "vfredusum.vs v8, v20, v8 \n\t"
            : // No C++ outputs
            : [vl] "r"(vl),
              [lhs_ptr] "r"(lhs_data + k),
              [rhs_ptr] "r"(rhs_data + k)
            : "v12", "v16", "v20" // Tell the compiler we trashed these temporaries
        );

        k += vl;
      }

      // Store
      asm volatile("vsetivli zero, 1, e32, m1, ta, ma;"
                   "vse32.v v8, (%0);"
                   :
                   : "r"(result_row + c)
                   : "memory"
      );
    }
  }
}

int main(int argc, char** argv) {
  MatMulF(kLhsRows, kInner, kRhsCols, lhs_input, rhs_input, result_output);
  return 0;
}
