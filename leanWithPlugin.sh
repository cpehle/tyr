MY_DIR=`dirname $0`
LEAN_LIB_PATH=`lean --print-libdir`
LD_PRELOAD="${MY_DIR}/plugin/build/TyrPlugin.dll ${LEAN_LIB_PATH}/libleanshared.so" LEAN_PATH=${MY_DIR}/build lean --plugin ${MY_DIR}/plugin/build/TyrPlugin.dll "$@"
