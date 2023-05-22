
#include "backprop_tools_adapter.h"

#include <backprop_tools/operations/arm.h>
#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
#include <backprop_tools/nn/layers/dense/operations_arm/dsp.h>
#include <backprop_tools/nn_models/mlp/operations_generic.h>
#include "data/test_backprop_tools_nn_models_mlp_persist_code.h"
#include "data/test_backprop_tools_nn_models_mlp_evaluation.h"


namespace bpt = backprop_tools;

using DEV_SPEC = bpt::devices::DefaultARMSpecification;
using DEVICE = bpt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using TI = typename mlp_1::SPEC::TI;
using DTYPE = typename mlp_1::SPEC::T;

DTYPE input_layer_output_memory[mlp_1::SPEC::OUTPUT_DIM];
bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> input_layer_output = {(DTYPE*)input_layer_output_memory};

void backprop_tools_run(float* output_mem){
    DTYPE buffer_tick_memory[mlp_1::SPEC::HIDDEN_DIM];
    DTYPE buffer_tock_memory[mlp_1::SPEC::HIDDEN_DIM];
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tick = {(DTYPE*)buffer_tick_memory};
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::HIDDEN_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tock = {(DTYPE*)buffer_tock_memory};

    decltype(mlp_1::mlp)::template Buffers<1> buffers = {buffer_tick, buffer_tock};
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, mlp_1::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
    auto input_sample = bpt::row(device, input::container, 0);
    int iterations = 10;
    for(int iteration_i = 0; iteration_i < iterations; iteration_i++){
        bpt::evaluate(device, mlp_1::mlp, input_sample, output, buffers);
    }
}