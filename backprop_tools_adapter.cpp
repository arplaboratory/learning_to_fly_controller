
#include "backprop_tools_adapter.h"

#include <backprop_tools/operations/arm.h>
#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
#include <backprop_tools/nn/layers/dense/operations_arm/dsp.h>
#include <backprop_tools/nn_models/mlp/operations_generic.h>
#include "data/actor_000000000500000.h"
// #include "data/test_backprop_tools_nn_models_mlp_evaluation.h"


namespace bpt = backprop_tools;

using DEV_SPEC = bpt::devices::DefaultARMSpecification;
using DEVICE = bpt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = decltype(actor::mlp);
using TI = typename ACTOR_TYPE::SPEC::TI;
using DTYPE = typename ACTOR_TYPE::SPEC::T;

void backprop_tools_run(float* input_mem, float* output_mem){
    DTYPE buffer_tick_memory[ACTOR_TYPE::SPEC::HIDDEN_DIM];
    DTYPE buffer_tock_memory[ACTOR_TYPE::SPEC::HIDDEN_DIM];
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::HIDDEN_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tick = {(DTYPE*)buffer_tick_memory};
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::HIDDEN_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> buffer_tock = {(DTYPE*)buffer_tock_memory};

    ACTOR_TYPE::template Buffers<1> buffers = {buffer_tick, buffer_tock};
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)output_mem};
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> input = {(DTYPE*)input_mem};
    int iterations = 1;
    for(int iteration_i = 0; iteration_i < iterations; iteration_i++){
        bpt::evaluate(device, actor::mlp, input, output, buffers);
    }
}