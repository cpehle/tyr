#!/bin/bash

TORCHDIR="../../thirdparty/libtorch"
INCPATH="-I$TORCHDIR/include -I$TORCHDIR/include/torch/csrc/api/include  -I$TORCHDIR/include/TH  -I$TORCHDIR/include/THC"
LIBPATH=$TORCHDIR/lib/
# INCPATH=`python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))"`
# LIBPATH=`python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')"`
USE_CUDA=`python -c "import torch; print(torch.cuda.is_available())"`
TORCH_LIBS=""

# echo $INCPATH
# echo $LIBPATH

mkdir -p build/
lean Torch.lean --c=build/Torch.cpp
c++ -std=c++14 -c -o build/Torch.o build/Torch.cpp `leanc -print-cflags`
c++ -std=c++14 -c -o build/ttorch.o ttorch.cpp `leanc -print-cflags` $INCPATH
c++ -std=c++14 -o torch build/Torch.o build/ttorch.o  -Wl,--no-as-needed -L$LIBPATH -ltorch -ltorch_cpu -lc10 `leanc -print-ldflags` 