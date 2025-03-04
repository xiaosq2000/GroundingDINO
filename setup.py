# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------------------------
# Modified from
# https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/setup.py
# https://github.com/facebookresearch/detectron2/blob/main/setup.py
# https://github.com/open-mmlab/mmdetection/blob/master/setup.py
# https://github.com/Oneflow-Inc/libai/blob/main/setup.py
# ------------------------------------------------------------------------------------------------

import glob
import os
import platform
import subprocess
import sys

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

# Platform check
if platform.system() != "Linux":
    print("Warning: This package is only designed for Linux systems.")
    print("It may not build correctly on your platform.")

# GroundingDINO version info
version = "0.1.0"
package_name = "groundingdino"
cwd = os.path.dirname(os.path.abspath(__file__))


def setup_cuda_env():
    """Setup CUDA environment by finding nvcc executable."""
    if "CUDA_HOME" in os.environ:
        return True

    # Try to find CUDA location via nvcc
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"], text=True).strip()
        if nvcc_path:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
            os.environ["CUDA_HOME"] = cuda_home
            print(f"Auto-detected CUDA_HOME: {cuda_home}")
            return True
    except subprocess.SubprocessError:
        print("Warning: nvcc not found in PATH.")
        return False


def write_version_file():
    version_path = os.path.join(cwd, "groundingdino", "version.py")
    with open(version_path, "w") as f:
        f.write(f"__version__ = '{version}'\n")

        # Try to get git version
        sha = "Unknown"
        try:
            if os.path.exists(os.path.join(cwd, ".git")):
                sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode("ascii").strip()
        except Exception:
            pass

        f.write(f"git_version = '{sha}'\n")


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )

    sources = [main_source] + sources

    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Try to setup CUDA environment
    setup_cuda_env()

    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print(f"Compiling with CUDA support. CUDA_HOME = {CUDA_HOME}")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling without CUDA. Either CUDA_HOME is not set or PyTorch was not built with CUDA support.")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        # Return empty extensions for CPU-only mode
        return []

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file."""
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


if __name__ == "__main__":
    print(f"Building wheel {package_name}-{version}")

    setup_success = False

    try:
        write_version_file()

        # Read requirements or use defaults
        requirements = []
        if os.path.exists("requirements.txt"):
            requirements = parse_requirements("requirements.txt")
        else:
            requirements = ["torch>=1.7.0", "torchvision>=0.8.0"]

        # Setup the package
        setup(
            name="groundingdino",
            version=version,
            author="International Digital Economy Academy, Shilong Liu",
            url="https://github.com/IDEA-Research/GroundingDINO",
            description="open-set object detector",
            packages=find_packages(exclude=("configs", "tests")),
            ext_modules=get_extensions(),
            cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
            install_requires=requirements,
            python_requires=">=3.7",
        )

        # Try importing to verify success
        try:
            import groundingdino

            try:
                import groundingdino._C

                print("Successfully built GroundingDINO with CUDA support!")
            except ImportError:
                print("Successfully built GroundingDINO (CPU-only mode)")
            setup_success = True
        except ImportError:
            print("Warning: GroundingDINO was installed but cannot be imported.")

    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback

        traceback.print_exc()

    if not setup_success:
        print("------------------------------------------------")
        print("GroundingDINO setup did not complete successfully.")
        print("Please check your CUDA and PyTorch installation.")
        print("If you want to use CPU-only mode, make sure to unset CUDA_HOME.")
        print("------------------------------------------------")
        sys.exit(1)
