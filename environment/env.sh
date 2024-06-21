deactivate
module purge

# Note that the modules loaded on the login node differ from compute nodes.
# Use `module avail` on the login node to see modules that can be loaded to fix errors.
module load gnu12/12.2.0

module load \
	linux-rocky8-x86_64/gcc/12.2.0/python/3.11.1-efo4f5x \
	linux-rocky8-x86_64/gcc/12.2.0/swig/4.1.1-qarng36 \
	linux-rocky8-x86_64/gcc/12.2.0/openblas/0.3.21-zy2y7i4 \
	linux-rocky8-x86_64/gcc/12.2.0/hdf5/1.12.1-cdxzvhd

# Other optional modules -- can remove if undesired
module load \
	linux-rocky8-x86_64/gcc/12.2.0/git/2.39.1-66kunjv \
	linux-rocky8-x86_64/gcc/12.2.0/vim/9.0.0045-rtvvtu6

# Ensure this is correct.
# Directories should match loaded modules
# - gcc versions
# - versions and (start of) hashes of openblas, hdf5 and openmpi
# `ls $BLAS_INCDIR` should show files like `cblas.h f77blas.h lapack.h ...`
# `ls $HDF5_BASE` should show directories like "bin include lib share"

# Might be that not all of these are required to install and use AMICI.
PATH_TO_OPENBLAS="/opt/ohpc/pub/spack/develop/opt/spack/linux-rocky8-zen2/gcc-12.2.0/openblas-0.3.21-zy2y7i4qybyoxlh6itcvjgiruhsimi7b"
export BLAS_INCDIR="$PATH_TO_OPENBLAS/include/"
export BLAS_CFLAGS="-I$BLAS_INCDIR"
export BLAS_LIBS="-lopenblas"
export HDF5_BASE="/opt/ohpc/pub/spack/develop/opt/spack/linux-rocky8-zen2/gcc-12.2.0/hdf5-1.12.1-cdxzvhdnb2hdcb4ib3aoifzqikbjzly3"
export LDFLAGS="-L$PATH_TO_OPENBLAS/lib/ $LDFLAGS"
export LD_LIBRARY_PATH="$HDF5_BASE/lib:$PATH_TO_OPENBLAS/lib:$LD_LIBRARY_PATH"

# Edit this command to activate your virtual environment.
source /home/user/nUDE/venv/bin/activate
