# Copyright 2026 Google LLC
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

# Set the maximum number of threads for Vivado to use.
puts "**************************************************"
puts "*** Starting vivado_general_build_settings.tcl ***"
puts "**************************************************"
puts "Setting general.maxThreads 32"
set targetMaxThreads 32
set_param general.maxThreads $targetMaxThreads
set maxThreads [get_param general.maxThreads]

# Print the actual value and verify if it matches the target
if {$maxThreads == $targetMaxThreads} {
    puts "Success: Max Threads is correctly set to $maxThreads."
} else {
    puts "Warning: Target maxThreads was $targetMaxThreads, but Vivado set it to $maxThreads."
}

puts "**************************************************"
puts "*** Completed vivado_general_build_settings.tcl ***"
puts "**************************************************"
