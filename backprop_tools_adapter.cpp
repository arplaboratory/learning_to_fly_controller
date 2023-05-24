
#include "backprop_tools_adapter.h"

#include <backprop_tools/operations/arm.h>
#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
#include <backprop_tools/nn/layers/dense/operations_arm/dsp.h>
#include <backprop_tools/nn_models/mlp/operations_generic.h>
// #include "data/actor_000000000500000.h"
#include "data/actor_000000001000000.h"
// #include "data/actor_000000000050000.h"
// #include "data/actor_000000000000000.h"
// #include "data/test_backprop_tools_nn_models_mlp_evaluation.h"


namespace bpt = backprop_tools;

using DEV_SPEC = bpt::devices::DefaultARMSpecification;
using DEVICE = bpt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = decltype(bpt::checkpoint::actor::mlp);
using TI = typename ACTOR_TYPE::SPEC::TI;
using DTYPE = typename ACTOR_TYPE::SPEC::T;


static inline void observe_rotation_matrix(const bpt::Matrix<bpt::matrix::Specification<DTYPE, TI, 1, 13>>& state, bpt::Matrix<bpt::matrix::Specification<DTYPE, TI, 1, 18>>& observation){
    float qw = get(state, 0, 3);
    float qx = get(state, 0, 4);
    float qy = get(state, 0, 5);
    float qz = get(state, 0, 6);
    set(observation, 0,  0 + 0, get(state, 0, 0));
    set(observation, 0,  0 + 1, get(state, 0, 1));
    set(observation, 0,  0 + 2, get(state, 0, 2));
    set(observation, 0,  3 + 0, (1 - 2*qy*qy - 2*qz*qz));
    set(observation, 0,  3 + 1, (    2*qx*qy - 2*qw*qz));
    set(observation, 0,  3 + 2, (    2*qx*qz + 2*qw*qy));
    set(observation, 0,  3 + 3, (    2*qx*qy + 2*qw*qz));
    set(observation, 0,  3 + 4, (1 - 2*qx*qx - 2*qz*qz));
    set(observation, 0,  3 + 5, (    2*qy*qz - 2*qw*qx));
    set(observation, 0,  3 + 6, (    2*qx*qz - 2*qw*qy));
    set(observation, 0,  3 + 7, (    2*qy*qz + 2*qw*qx));
    set(observation, 0,  3 + 8, (1 - 2*qx*qx - 2*qy*qy));
    set(observation, 0, 12 + 0, get(state, 0, 3 + 4 + 0));
    set(observation, 0, 12 + 1, get(state, 0, 3 + 4 + 1));
    set(observation, 0, 12 + 2, get(state, 0, 3 + 4 + 2));
    set(observation, 0, 15 + 0, get(state, 0, 3 + 4 + 3 + 0));
    set(observation, 0, 15 + 1, get(state, 0, 3 + 4 + 3 + 1));
    set(observation, 0, 15 + 2, get(state, 0, 3 + 4 + 3 + 2));
}

void backprop_tools_control_rotation_matrix(float* state, float* actions){
    // static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == 18);
    // bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, 13>> state_matrix = {(DTYPE*)state}; 
    // // observe_rotation_matrix(state_matrix, input);
    // bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)actions};
    // bpt::evaluate(device, bpt::checkpoint::actor::mlp, input, output, buffers);
}

// DTYPE buffer_tick_memory[ACTOR_TYPE::SPEC::HIDDEN_DIM];
// DTYPE buffer_tock_memory[ACTOR_TYPE::SPEC::HIDDEN_DIM];
// DTYPE buffer_input[ACTOR_TYPE::SPEC::INPUT_DIM];

static ACTOR_TYPE::template Buffers<1, bpt::MatrixStaticTag> buffers;
static bpt::MatrixStatic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> input;
static bpt::MatrixStatic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output;

void backprop_tools_init(){
    bpt::malloc(device, buffers);
    bpt::malloc(device, input);
    bpt::malloc(device, output);
}

float backprop_tools_test(float* output_mem){
    observe_rotation_matrix(bpt::checkpoint::state::container, input);
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, input, output, buffers);
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(bpt::get(output, 0, i) - bpt::get(bpt::checkpoint::action::container, 0, i));
        output_mem[i] = bpt::get(bpt::checkpoint::action::container, 0, i);
    }
    return acc;
}