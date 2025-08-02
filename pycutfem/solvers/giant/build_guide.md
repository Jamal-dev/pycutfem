**Run:
# Clean
rm -f *.o *.a

# Compile each GIANT source to position-independent objects
gfortran -c -O3 -fPIC -ffixed-form -ffixed-line-length-none -std=legacy -fallow-argument-mismatch \
  giant.f easypack_giant.f gmres_giant.f linalg_giant.f linpack_giant.f zibconst.f zibsec.f zibmon.f

# Archive them into a static library
ar rcs libgiant_pic.a giant.o easypack_giant.o gmres_giant.o linalg_giant.o \
    linpack_giant.o zibconst.o zibsec.o zibmon.o
ranlib libgiant_pic.a

rm -rf build *.so *.pyd __pycache__ 2>/dev/null
export FC=gfortran
export NPY_DISTUTILS_APPEND_FLAGS=1

python -m numpy.f2py -c -m giant_solver \
  giant_shim.f90 ./libgiant_pic.a \
  --fcompiler=gnu95 --opt='-O3' --f90flags='-fallow-argument-mismatch'
