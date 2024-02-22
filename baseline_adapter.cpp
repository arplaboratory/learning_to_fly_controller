
#include "rl_tools_adapter.h"

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/dense/operations_arm/dsp.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor_baseline.h"

#define RL_TOOLS_CONTROL_STATE_ROTATION_MATRIX
// #define RL_TOOLS_DISABLE_TEST


// Definitions
namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = actor::MODEL;
using TI = typename ACTOR_TYPE::SPEC::TI;
using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI CONTROL_FREQUENCY_MULTIPLE = 5;
static TI controller_tick = 0;
constexpr TI HISTORY_LENGTH = 2; //rlt::checkpoint::environment::ACTION_HISTORY_LENGTH
constexpr TI INPUT_DIM_STATE = 13;
constexpr TI INPUT_DIM_ACTION = 4;
constexpr TI INPUT_DIM_PER_STEP = INPUT_DIM_STATE + INPUT_DIM_ACTION; //rlt::checkpoint::environment::ACTION_HISTORY_LENGTH
static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == (INPUT_DIM_PER_STEP * HISTORY_LENGTH));
static bool initialized = false;

// State
static ACTOR_TYPE::template Buffer<1, rlt::MatrixStaticTag> buffers;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input_history;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input_buffer;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, INPUT_DIM_PER_STEP>> current_input;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM>> output;
#ifdef RL_TOOLS_ACTION_HISTORY
static T action_history[ACTION_HISTORY_LENGTH][ACTOR_TYPE::SPEC::OUTPUT_DIM];
#endif


// Helper functions (without side-effects)
template <typename STATE_SPEC, typename OBS_SPEC>
static inline void observe(const rlt::Matrix<STATE_SPEC>& state, rlt::Matrix<OBS_SPEC>& observation){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == INPUT_DIM_STATE);
    float qw = rlt::get(state, 0, 3);
    float qx = rlt::get(state, 0, 4);
    float qy = rlt::get(state, 0, 5);
    float qz = rlt::get(state, 0, 6);
    if(qw < 0){
        qw = -qw;
        qx = -qx;
        qy = -qy;
        qz = -qz;
    }
    rlt::set(observation, 0,  0 + 0, rlt::get(state, 0, 0));
    rlt::set(observation, 0,  0 + 1, rlt::get(state, 0, 1));
    rlt::set(observation, 0,  0 + 2, rlt::get(state, 0, 2) + 1.0); // in hover mode the policy tries to go to [0, 0, 1]
    rlt::set(observation, 0,  3 + 0, qx);
    rlt::set(observation, 0,  3 + 1, qy);
    rlt::set(observation, 0,  3 + 2, qz);
    rlt::set(observation, 0,  3 + 3, qw);
    rlt::set(observation, 0,  7 + 0, rlt::get(state, 0, 3 + 4 + 0));
    rlt::set(observation, 0,  7 + 1, rlt::get(state, 0, 3 + 4 + 1));
    rlt::set(observation, 0,  7 + 2, rlt::get(state, 0, 3 + 4 + 2));
    rlt::set(observation, 0, 10 + 0, rlt::get(state, 0, 3 + 4 + 3 + 0));
    rlt::set(observation, 0, 10 + 1, rlt::get(state, 0, 3 + 4 + 3 + 1));
    rlt::set(observation, 0, 10 + 2, rlt::get(state, 0, 3 + 4 + 3 + 2));
    // rlt::set(observation, 0, 13 + 0, rlt::get(action, 0, 0));
    // rlt::set(observation, 0, 13 + 1, rlt::get(action, 0, 1));
    // rlt::set(observation, 0, 13 + 2, rlt::get(action, 0, 2));
    // rlt::set(observation, 0, 13 + 2, rlt::get(action, 0, 3));
}

// Main functions (possibly with side effects)
void rl_tools_init(){
    rlt::malloc(device, buffers);
    rlt::malloc(device, input_history);
    rlt::malloc(device, input_buffer);
    rlt::malloc(device, current_input);
    rlt::malloc(device, output);
    rlt::set_all(device, output, 0);
    initialized = false;
#ifdef RL_TOOLS_ACTION_HISTORY
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
            action_history[step_i][action_i] = 0;
        }
    }
#endif
    controller_tick = 0;
}

char* rl_tools_get_checkpoint_name(){
    return "baseline"; 
}

float rl_tools_test(float* output_mem){
#ifndef RL_TOOLS_DISABLE_TEST
    // rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input;
    // rlt::malloc(device, input);
    for(TI input_i=0; input_i < ACTOR_TYPE::SPEC::INPUT_DIM; input_i++){
        T mean = rlt::get(observation_mean::container, 0, input_i);
        T std = rlt::get(observation_std::container, 0, input_i);
        T input_value = 0;
        if(input_i == 2){
            input_value = 1.0;
        }
        rlt::set(input_buffer, 0, input_i, (input_value - mean) / std);
    }

    rlt::evaluate(device, actor::model, input_buffer, output, buffers);
    // float acc = 0;
    // observation_mean::container._data = observation_mean::memory;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        // acc += std::abs(rlt::get(output, 0, i) - rlt::get(rlt::checkpoint::action::container, 0, i));
        output_mem[i] = rlt::get(output, 0, i);
    }
    return 0; //acc;
#else
    return 0;
#endif
}


void rl_tools_control(float* state, float* actions){
    if(!initialized){
        rlt::set_all(device, input_history, 0);
        initialized = true;
    }
    for(TI step_i = 0; step_i < HISTORY_LENGTH - 1; step_i++){
        auto current_step_source = rlt::view(device, input_history, rlt::matrix::ViewSpec<1, INPUT_DIM_PER_STEP>{}, 0, (step_i+1)*INPUT_DIM_PER_STEP);
        auto current_step_target = rlt::view(device, input_history, rlt::matrix::ViewSpec<1, INPUT_DIM_PER_STEP>{}, 0, step_i*INPUT_DIM_PER_STEP);
        rlt::copy(device, device, current_step_source, current_step_target);
    }
    auto last_action = rlt::view(device, input_history, rlt::matrix::ViewSpec<1, INPUT_DIM_ACTION>{}, 0, (HISTORY_LENGTH-1)*INPUT_DIM_PER_STEP + INPUT_DIM_STATE);
    rlt::copy(device, device, output, last_action);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 13, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(T*)state}; 
    auto last_step_input = rlt::view(device, input_history, rlt::matrix::ViewSpec<1, INPUT_DIM_STATE>{}, 0, (HISTORY_LENGTH-1)*INPUT_DIM_PER_STEP);
    observe(state_matrix, last_step_input);
    for(TI input_i=0; input_i < ACTOR_TYPE::SPEC::INPUT_DIM; input_i++){
        T value = rlt::get(input_history, 0, input_i);
        T mean = rlt::get(observation_mean::container, 0, input_i);
        T std = rlt::get(observation_std::container, 0, input_i);
        T normalized = (value - mean) / std;
        rlt::set(input_buffer, 0, input_i, normalized);
    }
    // rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)actions};
    rlt::evaluate(device, actor::model, input_buffer, output, buffers);
    for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
        T thrust = rlt::get(output, 0, action_i);
        T clipped_thrust = thrust < -1.0 ? -1.0 : (thrust > 1.0 ? 1.0 : thrust);
        T normed_thrust = (clipped_thrust + 1)/2;
        T normed_rpm = rlt::math::sqrt(device.math, normed_thrust);
        actions[action_i] = normed_rpm * 2.0 - 1.0;
    }
}
// void rl_tools_control_other(float* state, float* actions){
//     int substep = controller_tick % CONTROL_FREQUENCY_MULTIPLE;
//     rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 13, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(T*)state}; 
//     auto last_step_input = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_STATE>{}, 0, (HISTORY_LENGTH-1)*INPUT_DIM_PER_STEP);
//     observe(state_matrix, last_step_input);
//     if(substep == 0){
//         for(TI step_i = 0; step_i < HISTORY_LENGTH - 1; step_i++){
//             auto current_step_source = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_PER_STEP>{}, 0, (step_i+1)*INPUT_DIM_PER_STEP);
//             auto current_step_target = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_PER_STEP>{}, 0, step_i*INPUT_DIM_PER_STEP);
//             rlt::copy(device, device, current_step_source, current_step_target);
//         }
//         auto current_action = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_ACTION>{}, 0, (HISTORY_LENGTH-1)*INPUT_DIM_PER_STEP + INPUT_DIM_STATE);
//         rlt::copy(device, device, output, current_action);
//     }
//     if(!initialized){
//         auto current_action = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_ACTION>{}, 0, (HISTORY_LENGTH-1)*INPUT_DIM_PER_STEP + INPUT_DIM_STATE);
//         rlt::set_all(device, current_action, 0);
//         auto source = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_PER_STEP>{}, 0, (HISTORY_LENGTH-1)*INPUT_DIM_PER_STEP);
//         for(TI step_i = 0; step_i < HISTORY_LENGTH-1; step_i++){
//             auto current_step_target = rlt::view(device, input, rlt::matrix::ViewSpec<1, INPUT_DIM_PER_STEP>{}, 0, step_i*INPUT_DIM_PER_STEP);
//             rlt::copy(device, device, source, current_step_target);
//         }
//         initialized = true;
//     }
// #ifdef RL_TOOLS_ACTION_HISTORY
//     auto action_history_observation = rlt::view(device, input, rlt::matrix::ViewSpec<1, ACTION_HISTORY_LENGTH * ACTOR_TYPE::SPEC::OUTPUT_DIM>{}, 0, 18);
//     for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
//         for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
//             rlt::set(action_history_observation, 0, step_i * ACTOR_TYPE::SPEC::OUTPUT_DIM + action_i, action_history[step_i][action_i]);
//         }
//     }
// #endif
//     for(TI input_i=0; input_i < ACTOR_TYPE::SPEC::INPUT_DIM; input_i++){
//         T value = rlt::get(input, 0, input_i);
//         T mean = rlt::get(observation_mean::container, 0, input_i);
//         T std = rlt::get(observation_std::container, 0, input_i);
//         T normalized = (value - mean) / std;
//         rlt::set(input, 0, input_i, normalized);
//     }
//     rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)actions};
//     rlt::evaluate(device, actor::model, input, output, buffers);
//     for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
//         T clipped_thrust = rlt::get(output, 0, action_i) < -1.0 ? -1.0 : (rlt::get(output, 0, action_i) > 1.0 ? 1.0 : rlt::get(output, 0, action_i));
//         T normed_thrust = (clipped_thrust + 1)/2;
//         T normed_rpm = rlt::math::sqrt(device.math, normed_thrust);
//         set(output, 0, action_i, normed_rpm * 2.0 - 1.0);
//     }
//     // if(substep == 0){
//     //     for(TI step_i = 0; step_i < HISTORY_LENGTH - 1; step_i++){
//     //         for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
//     //             action_history[step_i][action_i] = action_history[step_i + 1][action_i];
//     //         }
//     //     }
//     // }
//     // for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
//     //     T value = action_history[HISTORY_LENGTH - 1][action_i];
//     //     value *= substep;
//     //     value += rlt::get(output, 0, action_i);
//     //     value /= substep + 1;
//     //     action_history[ACTION_HISTORY_LENGTH - 1][action_i] = value;
//     // }
//     controller_tick++;
// }
