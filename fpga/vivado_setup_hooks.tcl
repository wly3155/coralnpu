# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Elevate warning for not finding file for readmemh to ERROR.
set_msg_config -id {[Synth 8-4445]} -new_severity ERROR

set workroot [pwd]

# If we see the Xilinx DDR core, register the post-bitstream hook to stitch the calibration FW.
if {[file exists "${workroot}/src/xilinx_ddr4_0_0.1_0"]} {
    set_property STEPS.WRITE_BITSTREAM.TCL.POST "${workroot}/vivado_hook_write_bitstream_post.tcl" [get_runs impl_1]
}

# Register pre-synthesis and pre-implementation hooks.
set_property STEPS.SYNTH_DESIGN.TCL.PRE "${workroot}/vivado_hook_synthesis_pre.tcl" [get_runs synth_1]
set_property STEPS.OPT_DESIGN.TCL.PRE "${workroot}/vivado_hook_implementation_pre.tcl" [get_runs impl_1]

# Enable ultrathreads for placement and routing via MORE OPTIONS (appending to existing options)
puts "Adding -ultrathreads to PLACE_DESIGN and ROUTE_DESIGN MORE OPTIONS"
try {
    foreach step {PLACE_DESIGN ROUTE_DESIGN} {
        set opts [get_property "STEPS.${step}.ARGS.MORE OPTIONS" [get_runs impl_1]]
        if {![string match "*-ultrathreads*" $opts]} {
            set_property "STEPS.${step}.ARGS.MORE OPTIONS" "$opts -ultrathreads" [get_runs impl_1]
        }
    }
} on error {err} {
    puts "WARNING: Failed to set -ultrathreads property. Build will continue without this optimization."
    puts "Error message: $err"
}

# Readback and verify
set place_opt_verify [get_property {STEPS.PLACE_DESIGN.ARGS.MORE OPTIONS} [get_runs impl_1]]
set route_opt_verify [get_property {STEPS.ROUTE_DESIGN.ARGS.MORE OPTIONS} [get_runs impl_1]]
puts "Readback: PLACE_DESIGN.ARGS.MORE OPTIONS is '$place_opt_verify'"
puts "Readback: ROUTE_DESIGN.ARGS.MORE OPTIONS is '$route_opt_verify'"

