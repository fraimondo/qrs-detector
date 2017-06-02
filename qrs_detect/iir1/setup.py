from distutils.core import setup, Extension
import platform

include = ['./c']
extra_compile_args=['-Wwrite-strings']
system = platform.system()
if system == 'Linux':
    extra_link_args = ['-fPIC', '-lstdc++']
    extra_compile_args.append('-fPIC')
    extra_compile_args.append('-lstdc++')
elif system == 'Darwin':
    extra_link_args = []

extra_link_args.append('c/.libs/libiir.a')

module1 = Extension('pyiir1',
                    sources=['pyiir1.c', 'butterworth.c'],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    include_dirs=include)

setup(name='Pyiir1',
      version='1.0',
      description='This is the IIR1 bindings to python interface',
      ext_modules=[module1])
