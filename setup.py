from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, include_paths, library_paths
import glob
import os


def _getenv_or_die(env_name: str):
    env = os.getenv(env_name)
    if env is None:
        raise ValueError(f"{env_name} is not set")
    return env


def get_ext():
    ext_name = "dipu_ext.ext_"
    # 包含所有算子文件
    op_files = glob.glob("./ext_op/*.cpp")
    include_dirs = []
    system_include_dirs = include_paths()
    define_macros = []
    extra_objects = []
    library_dirs = library_paths()
    libraries = ["c10", "torch", "torch_cpu", "torch_python"]
    extra_link_args = []

    dipu_root = _getenv_or_die("DIPU_ROOT")
    diopi_path = _getenv_or_die("DIOPI_PATH")
    vendor_include_dirs = _getenv_or_die("VENDOR_INCLUDE_DIRS")
    nccl_include_dirs = os.getenv("NCCL_INCLUDE_DIRS")  # nv所需
    system_include_dirs += [
        dipu_root,
        os.path.join(dipu_root, "dist/include"),
        os.path.join(diopi_path, "include"),
        vendor_include_dirs,
    ]
    if nccl_include_dirs:
        system_include_dirs.append(nccl_include_dirs)
    library_dirs += [dipu_root]
    libraries += ["torch_dipu"]

    extra_compile_args = {"cxx": []}
    extra_compile_args["cxx"] = ["-std=c++14"]
    extra_compile_args["cxx"] += ["-isystem" + path for path in system_include_dirs]
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


setup(name="dipu_ext", ext_modules=get_ext(), cmdclass={"build_ext": BuildExtension})
