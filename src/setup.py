from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules=cythonize(["./support_function/utils.pyx",
                          "./support_function/model_router.pyx"]))
