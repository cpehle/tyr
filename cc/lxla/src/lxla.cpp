#include <lean/lean.h>

#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/lu_decomposition.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/primitive_util.h"


void new_builder() {
    char* name = "test";
    xla::XlaBuilder* builder = new xla::XlaBuilder(name);
}

void create_subbuilder() {
}

void build() {
    xla::XlaBuilder** builder;
    xla::XlaOp* root;
}


void parameter() {
    xla::XlaBuilder** builder;
    int64_t param_num;
    xla::Shape* shape;
    std::string name;
}

// 


// shape functions

// xla ops

// tuples

// conditionals

// slice




// binary ops
void xla_binary_op() {
    xla::XlaOp *lhs;
    xla::XlaOp *rhs;
    std::vector<uint64_t> broadcast_dims;
}

// unary ops