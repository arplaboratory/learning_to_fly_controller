
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
// #define BACKPROP_TOOLS_DISABLE_TEST
#define BACKPROP_TOOLS_ACTION_HISTORY


// Definitions
namespace bpt = backprop_tools;

using DEV_SPEC = bpt::devices::DefaultARMSpecification;
using DEVICE = bpt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = decltype(bpt::checkpoint::actor::mlp);
using TI = typename ACTOR_TYPE::SPEC::TI;
using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI CONTROL_FREQUENCY_MULTIPLE = 1;
static TI controller_tick = 0;

// State
static ACTOR_TYPE::template Buffers<1, bpt::MatrixStaticTag> buffers;
static bpt::MatrixStatic<bpt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input;
static bpt::MatrixStatic<bpt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM>> output;
#ifdef BACKPROP_TOOLS_ACTION_HISTORY
static T action_history[bpt::checkpoint::environment::ACTION_HISTORY_LENGTH][ACTOR_TYPE::SPEC::OUTPUT_DIM];
#endif


// Helper functions (without side-effects)
template <typename STATE_SPEC, typename OBS_SPEC>
static inline void observe_rotation_matrix(const bpt::Matrix<STATE_SPEC>& state, bpt::Matrix<OBS_SPEC>& observation){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == 18);
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
#ifdef BACKPROP_TOOLS_ACTION_HISTORY
    for(TI step_i = 0; step_i < bpt::checkpoint::environment::ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
            action_history[step_i][action_i] = 0;
        }
    }
#endif
    controller_tick = 0;
}

float backprop_tools_test(float* output_mem){
#ifndef BACKPROP_TOOLS_DISABLE_TEST
// #ifndef BACKPROP_TOOLS_CONTROL_STATE_QUATERNION
//     auto state = bpt::view(device, bpt::checkpoint::state::container, bpt::matrix::ViewSpec<1, 13>{}, 0, 0);
//     // observe_rotation_matrix(state, input);
//     // bpt::evaluate(device, bpt::checkpoint::actor::mlp, input, output, buffers);
//     bpt::evaluate(device, bpt::checkpoint::actor::mlp, bpt::checkpoint::observation::container, output, buffers);
//     float acc = 0;
//     for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
//         acc += std::abs(bpt::get(output, 0, i) - bpt::get(bpt::checkpoint::action::container, 0, i));
//         output_mem[i] = bpt::get(bpt::checkpoint::action::container, 0, i);
//     }
//     return acc;
// #elif defined(BACKPROP_TOOLS_CONTROL_STATE_ROTATION_MATRIX)
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, bpt::checkpoint::observation::container, output, buffers);
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(bpt::get(output, 0, i) - bpt::get(bpt::checkpoint::action::container, 0, i));
        output_mem[i] = bpt::get(bpt::checkpoint::action::container, 0, i);
    }
    return acc;
// #endif
#else
    return 0;
#endif
}

#ifdef BACKPROP_TOOLS_CONTROL_STATE_QUATERNION
void backprop_tools_control(float* state, float* actions){
    static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == 13);
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 13, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(T*)state}; 
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)actions};
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, state_matrix, output, buffers);
    controller_tick++;
}
#elif defined(BACKPROP_TOOLS_CONTROL_STATE_ROTATION_MATRIX)
void backprop_tools_control(float* state, float* actions){
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 13, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(T*)state}; 
    auto state_rotation_matrix_input = bpt::view(device, input, bpt::matrix::ViewSpec<1, 18>{}, 0, 0);
    observe_rotation_matrix(state_matrix, state_rotation_matrix_input);
#ifdef BACKPROP_TOOLS_ACTION_HISTORY
    auto action_history_observation = bpt::view(device, input, bpt::matrix::ViewSpec<1, bpt::checkpoint::environment::ACTION_HISTORY_LENGTH * ACTOR_TYPE::SPEC::OUTPUT_DIM>{}, 0, 18);
    for(TI step_i = 0; step_i < bpt::checkpoint::environment::ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
            bpt::set(action_history_observation, 0, step_i * ACTOR_TYPE::SPEC::OUTPUT_DIM + action_i, action_history[step_i][action_i]);
        }
    }
#endif
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)actions};
    bpt::evaluate(device, bpt::checkpoint::actor::mlp, input, output, buffers);
#ifdef BACKPROP_TOOLS_ACTION_HISTORY
    if(controller_tick % CONTROL_FREQUENCY_MULTIPLE == 0){
        for(TI step_i = 0; step_i < bpt::checkpoint::environment::ACTION_HISTORY_LENGTH - 1; step_i++){
            for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
                action_history[step_i][action_i] = action_history[step_i + 1][action_i];
            }
        }
    }
    for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
        if(controller_tick % CONTROL_FREQUENCY_MULTIPLE == 0){
            action_history[bpt::checkpoint::environment::ACTION_HISTORY_LENGTH - 1][action_i] = 0;
        }
        action_history[bpt::checkpoint::environment::ACTION_HISTORY_LENGTH - 1][action_i] += bpt::get(output, 0, action_i) / ((T)CONTROL_FREQUENCY_MULTIPLE);
    }
#endif
    controller_tick++;
}
#endif
