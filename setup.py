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
    os.makedirs('dipu_ext', exist_ok=True)
    # 包含所有算子文件
    op_files = glob.glob("./ext_op/*.cpp")
    include_dirs = []
    system_include_dirs = include_paths()
    define_macros = []
    extra_objects = []
    library_dirs = library_paths()
    libraries = ["c10", "torch", "torch_cpu", "torch_python"]
    print(library_dirs)
    for i in library_dirs:
        extra_link_args = ['-Wl,-rpath,' + i]

    dipu_path = _getenv_or_die("DIPU_PATH")
    diopi_path = _getenv_or_die("DIOPI_PATH")
    torch_dipu_path = os.path.join(dipu_path, 'torch_dipu')
    dipu_lib_path = torch_dipu_path

    extra_link_args += ['-Wl,-rpath,' + dipu_lib_path]
    vendor_include_dirs = os.getenv("VENDOR_INCLUDE_DIRS")
    nccl_include_dirs = os.getenv("NCCL_INCLUDE_DIRS")  # nv所需
    system_include_dirs += [
        torch_dipu_path,
        os.path.join(torch_dipu_path, "dist/include"),
        os.path.join(diopi_path, "proto/include"),
    ]
    if vendor_include_dirs:
        system_include_dirs.append(vendor_include_dirs)
    if nccl_include_dirs:
        system_include_dirs.append(nccl_include_dirs)
    library_dirs += [dipu_lib_path]
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
