################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
import platform
from threading import Lock
import pycuda.driver as cuda
import pycuda.autoinit

guard_platform_info = Lock()

class PlatformInfo:
    def __init__(self):
        self.is_wsl_system = False
        self.wsl_verified = False
        self.is_integrated_gpu_system = False
        self.is_integrated_gpu_verified = False
        self.is_aarch64_platform = False
        self.is_aarch64_verified = False

    def is_wsl(self):
        with guard_platform_info:
            # Check if its already verified as WSL system or not.
            if not self.wsl_verified:
                try:
                    # Open /proc/version file
                    with open("/proc/version", "r") as version_file:
                        # Read the content
                        version_info = version_file.readline()
                        version_info = version_info.lower()
                        self.wsl_verified = True

                        # Check if "microsoft" is present in the version information
                        if "microsoft" in version_info:
                            self.is_wsl_system = True
                except Exception as e:
                    print(f"ERROR: Opening /proc/version failed: {e}")

        return self.is_wsl_system
    
    def is_integrated_gpu(self):
        # Using PyCUDA to check GPU type
        with guard_platform_info:
            if not self.is_integrated_gpu_verified:
                try:
                    import pycuda.driver as cuda
                    import pycuda.autoinit  # automatically initializes CUDA driver
    
                    device = cuda.Device(0)
                    
                    # Mimic 'properties.integrated' expected by DeepStream
                    class DummyProperties:
                        def __init__(self, device):
                            self.integrated = True if "Tegra" in device.name() or "Orin" in device.name() else False
    
                    properties = DummyProperties(device)
                    print("Is it Integrated GPU? :", properties.integrated)
                    self.is_integrated_gpu_system = properties.integrated
                    self.is_integrated_gpu_verified = True
    
                except Exception as e:
                    print("ERROR: PyCUDA failed:", e)
                    self.is_integrated_gpu_system = False
                    self.is_integrated_gpu_verified = True
    
        return self.is_integrated_gpu_system


    def is_platform_aarch64(self):
        #Check if platform is aarch64 using uname
        if not self.is_aarch64_verified:
            if platform.uname()[4] == 'aarch64':
                self.is_aarch64_platform =  True
            self.is_aarch64_verified = True
        return self.is_aarch64_platform

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
