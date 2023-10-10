from setuptools import setup, Extension
from torch.utils.cpp_extension import CppExtension, BuildExtension
import glob
import os


def get_ext():
    extensions = []
    extension = CppExtension
    ext_name = 'dipu_ext.ext_'
    # 包含所有算子文件
    op_files = glob.glob('./ext_op/*.cpp')
    include_dirs = [os.path.abspath('./ext_op')]
    define_macros = []
    extra_objects = []
    library_dirs = []
    libraries = []
    extra_link_args = []

    dipu_root = os.getenv('DIPU_ROOT')
    diopi_path = os.getenv('DIOPI_PATH')
    vendor_include_dirs = os.getenv('VENDOR_INCLUDE_DIRS')
    nccl_include_dirs = os.getenv('NCCL_INCLUDE_DIRS')     # nv所需
    include_dirs.append(dipu_root)
    include_dirs.append(dipu_root + "/dist/include")
    include_dirs.append(diopi_path + "/include")
    include_dirs.append(vendor_include_dirs)
    if nccl_include_dirs:
        include_dirs.append(nccl_include_dirs)
    library_dirs += [dipu_root]
    libraries += ['torch_dipu']

    extra_compile_args = {'cxx': []}
    extra_compile_args['cxx'] = ['-std=c++14']
    ext_ops = extension(
            name=ext_name,                # 拓展模块名字
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,  # 用于定义宏变量
            extra_objects=extra_objects,  # 传递object文件
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_link_args=extra_link_args)
    extensions.append(ext_ops)
    return extensions


setup(name='dipu_ext',
      ext_modules=get_ext(),
      cmdclass={'build_ext': BuildExtension})
