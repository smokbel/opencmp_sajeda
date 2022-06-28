module load StdEnv/2020 gcc/9.3.0 openmpi/4.0.3 python/3.8.2 mumps-metis/5.2.1 petsc/3.14.1 suitesparse/5.7.1 scipy-stack/2020b

export BASEDIR=$HOME/software/ngsuite
mkdir -p $BASEDIR
cd $BASEDIR && git clone https://github.com/NGSolve/ngsolve.git ngsolve-src
cd $BASEDIR/ngsolve-src && git checkout tags/v6.2.2102 && git submodule update --init --recursive
mkdir -p $BASEDIR/ngsolve-build
mkdir -p $BASEDIR/ngsolve-install
cd $BASEDIR && python3 -m venv .
source bin/activate
cd $BASEDIR/ngsolve-build

cmake -DCMAKE_INSTALL_PREFIX=${BASEDIR}/ngsolve-install -DCMAKE_CXX_FLAGS='-lpthread' -DUSE_GUI=OFF -DUSE_MKL=ON -DMKL_ROOT=${EBROOTIMKL}/mkl -DUSE_MPI=ON -DMETIS_DIR=${EBROOTMETIS} -DMETIS_INCLUDE_DIR=${EBROOTMETIS}/include -DUSE_UMFPACK=ON -DUSE_GUI=OFF -DUMFPACK_DIR=$EBROOTSUITESPARSE -DBUILD_UMFPACK=OFF -DINSTALL_DEPENDENCIES=OFF ${BASEDIR}/ngsolve-src

make -j8 && make install
echo "export NETGENDIR=${BASEDIR}/ngsolve-install/bin" >> $BASEDIR/bin/activate
echo "export PATH=\$NETGENDIR:\$PATH" >> $BASEDIR/bin/activate
export PYTHONPATH_TMP=`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib(1,0,''))"`
echo "export PYTHONPATH=\$NETGENDIR/../${PYTHONPATH_TMP}:\$PYTHONPATH" >> $BASEDIR/bin/activate

cd $BASEDIR && git clone https://github.com/NGSolve/ngs-petsc.git
mkdir -p $BASEDIR/ngs-petsc-build && cd $BASEDIR/ngs-petsc-build
cmake -DPETSC_EXECUTABLE_RUNS=YES $BASEDIR/ngs-petsc -DWITH_PETSC4PY=YES
make -j4
make install
