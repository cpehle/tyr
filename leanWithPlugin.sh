MY_DIR=`dirname $0`
LEAN_LIB_PATH=`lean --print-libdir`
export LD_LIBRARY_PATH=$MY_DIR/thirdparty/libtorch/lib
# LD_PRELOAD="${MY_DIR}/plugin/build/TyrPlugin.dll ${LEAN_LIB_PATH}/libleanshared.so" LEAN_PATH=${MY_DIR}/build lean --plugin ${MY_DIR}/plugin/build/TyrPlugin.dll "$@"
LEAN_PATH=${MY_DIR}/build lean --plugin ${MY_DIR}/plugin/build/TyrPlugin.dll "$@"
