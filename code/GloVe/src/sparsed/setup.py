from distutils.core import setup    
from distutils.extension import Extension                                     
from Cython.Build import cythonize

ext_modules = [
    Extension("corpus_cython",
              sources=["corpus_cython.pyx"],
              libraries=["m"]  # Unix-like specific
              ),
    Extension("glove_cython", 
              sources=["glove_cython.pyx"],
              libraries=["m"])
]

setup(
    name='cythonpack',
    ext_modules=cythonize(ext_modules),
)

# 命令行下执行：python setup.py build_ext --inplace
