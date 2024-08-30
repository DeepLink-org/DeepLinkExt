# Copyright (c) 2024, DeepLink.

from setuptools import find_packages, setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths, library_paths

import glob
import os
import subprocess


def _getenv_or_die(env_name: str):
    env = os.getenv(env_name)
    if env is None:
        raise ValueError(f"{env_name} is not set")
    return env


class BuildExtensionWithCompdb(BuildExtension):
    def build_extensions(self):
        super().build_extensions()
        try:
            self._gen_compdb()
        except Exception as e:
            print(f"Failed to generate compile_commands.json: {e}")

    def _gen_compdb(self):
        assert self.use_ninja
        build_ninja_file = glob.glob("./build/**/build.ninja", recursive=True)
        assert len(build_ninja_file) == 1
        with open("build/compile_commands.json", "w") as f:
            subprocess.run(
                ["ninja", "-f", build_ninja_file[0], "-t", "compdb"],
                stdout=f,
                check=True,
            )
        print("Generated build/compile_commands.json")


def get_ext():
    ext_name = "deeplink_ext.cpp_extensions"
    # 包含所有算子文件
    op_files = glob.glob("./csrc/*.cpp")
    include_dirs = []
    system_include_dirs = include_paths()
    define_macros = []
    extra_objects = []
    library_dirs = library_paths()
    libraries = ["c10", "torch", "torch_cpu", "torch_python"]
    extra_link_args = []

    dipu_root = _getenv_or_die("DIPU_ROOT")
    diopi_path = _getenv_or_die("DIOPI_PATH")
    vendor_include_dirs = os.getenv("VENDOR_INCLUDE_DIRS")
    nccl_include_dirs = os.getenv("NCCL_INCLUDE_DIRS")  # nv所需
    system_include_dirs += [
        dipu_root,
        os.path.join(dipu_root, "dist/include"),
        os.path.join(diopi_path, "include"),
    ]
    if vendor_include_dirs:
        system_include_dirs.append(vendor_include_dirs)
    if nccl_include_dirs:
        system_include_dirs.append(nccl_include_dirs)
    library_dirs += [dipu_root]
    libraries += ["torch_dipu"]

    extra_compile_args = ["-std=c++17", "-Wno-deprecated-declarations"]
    extra_compile_args += ["-isystem" + path for path in system_include_dirs]
    ext_ops = Extension(
        name=ext_name,  # 拓展模块名字
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,  # 用于定义宏变量
        extra_objects=extra_objects,  # 传递object文件
        extra_compile_args=extra_compile_args,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    )
    return [ext_ops]


setup(
    name="deeplink_ext",
    packages=find_packages(exclude=["build", "csrc", "tests"]),
    ext_modules=get_ext(),
    cmdclass={"build_ext": BuildExtensionWithCompdb},
    install_requires=["einops"],
)


setup(
    name='deeplink_ext_ops',
    ext_modules=[
        CppExtension(
            name='deeplink_ext_ops',
            sources=glob.glob("./csrc/*.cpp"),
            extra_compile_args=[' -g ']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })