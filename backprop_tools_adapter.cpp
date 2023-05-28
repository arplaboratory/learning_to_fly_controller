
#include "backprop_tools_adapter.h"

#include <backprop_tools/operations/arm.h>
#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
#include <backprop_tools/nn/layers/dense/operations_arm/dsp.h>
#include <backprop_tools/nn_models/mlp/operations_generic.h>
// #include "data/actor_000000000500000.h"
// #include "data/actor_000000001000000.h"
// #include "data/actor_000000004000000.h"
#include "data/actor.h"
// #include "data/test_backprop_tools_nn_models_mlp_evaluation.h"

// #define BACKPROP_TOOLS_CONTROL_STATE_QUATERNION
#define BACKPROP_TOOLS_CONTROL_STATE_ROTATION_MATRIX


// Definitions
namespace bpt = backprop_tools;

using DEV_SPEC = bpt::devices::DefaultARMSpecification;
using DEVICE = bpt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = decltype(bpt::checkpoint::actor::mlp);
using TI = typename ACTOR_TYPE::SPEC::TI;
using DTYPE = typename ACTOR_TYPE::SPEC::T;

// State
static ACTOR_TYPE::template Buffers<1, bpt::MatrixStaticTag> buffers;
static bpt::MatrixStatic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input;
static bpt::MatrixStatic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM>> output;


// Helper functions (without side-effects)
template <typename STATE_SPEC>
static inline void observe_rotation_matrix(const bpt::Matrix<STATE_SPEC>& state, bpt::Matrix<bpt::matrix::Specification<DTYPE, TI, 1, 18>>& observation){
    float qw = bpt::get(state, 0, 3);
    float qx = bpt::get(state, 0, 4);
    float qy = bpt::get(state, 0, 5);
    float qz = bpt::get(state, 0, 6);
    bpt::set(observation, 0,  0 + 0, bpt::get(state, 0, 0));
    bpt::set(observation, 0,  0 + 1, bpt::get(state, 0, 1));
    bpt::set(observation, 0,  0 + 2, bpt::get(state, 0, 2));
    bpt::set(observation, 0,  3 + 0, (1 - 2*qy*qy - 2*qz*qz));
    bpt::set(observation, 0,  3 + 1, (    2*qx*qy - 2*qw*qz));
    bpt::set(observation, 0,  3 + 2, (    2*qx*qz + 2*qw*qy));
    bpt::set(observation, 0,  3 + 3, (    2*qx*qy + 2*qw*qz));
    bpt::set(observation, 0,  3 + 4, (1 - 2*qx*qx - 2*qz*qz));
    bpt::set(observation, 0,  3 + 5, (    2*qy*qz - 2*qw*qx));
    bpt::set(observation, 0,  3 + 6, (    2*qx*qz - 2*qw*qy));
    bpt::set(observation, 0,  3 + 7, (    2*qy*qz + 2*qw*qx));
    bpt::set(observation, 0,  3 + 8, (1 - 2*qx*qx - 2*qy*qy));
    bpt::set(observation, 0, 12 + 0, bpt::get(state, 0, 3 + 4 + 0));
    bpt::set(observation, 0, 12 + 1, bpt::get(state, 0, 3 + 4 + 1));
    bpt::set(observation, 0, 12 + 2, bpt::get(state, 0, 3 + 4 + 2));
    bpt::set(observation, 0, 15 + 0, bpt::get(state, 0, 3 + 4 + 3 + 0));
    bpt::set(observation, 0, 15 + 1, bpt::get(state, 0, 3 + 4 + 3 + 1));
    bpt::set(observation, 0, 15 + 2, bpt::get(state, 0, 3 + 4 + 3 + 2));
}

// Main functions (possibly with side effects)
void backprop_tools_init(){
    bpt::malloc(device, buffers);
    bpt::malloc(device, input);
    bpt::malloc(device, output);
}

float backprop_tools_test(float* output_mem){
#ifndef BACKPROP_TOOLS_CONTROL_STATE_QUATERNION
    auto state = bpt::view(device, bpt::checkpoint::state::container, bpt::matrix::ViewSpec<1, 13>{}, 0, 0);
    observe_rotation_matrix(state, input);
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, input, output, buffers);
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(bpt::get(output, 0, i) - bpt::get(bpt::checkpoint::action::container, 0, i));
        output_mem[i] = bpt::get(bpt::checkpoint::action::container, 0, i);
    }
    return acc;
#elif defined(BACKPROP_TOOLS_CONTROL_STATE_ROTATION_MATRIX)
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, bpt::checkpoint::state::container, output, buffers);
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(bpt::get(output, 0, i) - bpt::get(bpt::checkpoint::action::container, 0, i));
        output_mem[i] = bpt::get(bpt::checkpoint::action::container, 0, i);
    }
    return acc;
#endif
}

#ifdef BACKPROP_TOOLS_CONTROL_STATE_QUATERNION
void backprop_tools_control(float* state, float* actions){
    static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == 13);
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, 13, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(DTYPE*)state}; 
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)actions};
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, state_matrix, output, buffers);
}
#elif defined(BACKPROP_TOOLS_CONTROL_STATE_ROTATION_MATRIX)
void backprop_tools_control(float* state, float* actions){
    static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == 18);
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, 13, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(DTYPE*)state}; 
    // bpt::set_all(device, state_matrix, 0);
    // bpt::set(state_matrix, 0, 0, 0.0);
    // bpt::set(state_matrix, 0, 1, 0.0);
    // bpt::set(state_matrix, 0, 2, 0.0);
    // bpt::set(state_matrix, 0, 3, 1.0);
    // bpt::set(state_matrix, 0, 4, 0.0);
    // bpt::set(state_matrix, 0, 5, 0.0);
    // bpt::set(state_matrix, 0, 6, 0.0);
    // bpt::set(state_matrix, 0, 7, 0.0);
    // bpt::set(state_matrix, 0, 8, 0.0);
    // bpt::set(state_matrix, 0, 9, 0.0);
    // bpt::set(state_matrix, 0, 10, 0.0);
    // bpt::set(state_matrix, 0, 11, 0.0);
    // bpt::set(state_matrix, 0, 12, 0.0);
    observe_rotation_matrix(state_matrix, input);
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(DTYPE*)actions};
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, input, output, buffers);
}
#endif
