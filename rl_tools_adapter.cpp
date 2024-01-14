
#include "rl_tools_adapter.h"

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/dense/operations_arm/dsp.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor.h"

#define RL_TOOLS_CONTROL_STATE_ROTATION_MATRIX
// #define RL_TOOLS_DISABLE_TEST
#define RL_TOOLS_ACTION_HISTORY


// Definitions
namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = rlt::checkpoint::actor::MODEL;
using TI = typename ACTOR_TYPE::SPEC::TI;
using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI CONTROL_FREQUENCY_MULTIPLE = 5;
static TI controller_tick = 0;
constexpr TI ACTION_HISTORY_LENGTH = 32; //rlt::checkpoint::environment::ACTION_HISTORY_LENGTH
#ifdef RL_TOOLS_ACTION_HISTORY
static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == (18 + ACTION_HISTORY_LENGTH * ACTOR_TYPE::SPEC::OUTPUT_DIM));
#else
static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == 18);
#endif

// State
static ACTOR_TYPE::template Buffer<1, rlt::MatrixStaticTag> buffers;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM>> output;
#ifdef RL_TOOLS_ACTION_HISTORY
static T action_history[ACTION_HISTORY_LENGTH][ACTOR_TYPE::SPEC::OUTPUT_DIM];
#endif


// Helper functions (without side-effects)
template <typename STATE_SPEC, typename OBS_SPEC>
static inline void observe_rotation_matrix(const rlt::Matrix<STATE_SPEC>& state, rlt::Matrix<OBS_SPEC>& observation){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == 18);
    float qw = rlt::get(state, 0, 3);
    float qx = rlt::get(state, 0, 4);
    float qy = rlt::get(state, 0, 5);
    float qz = rlt::get(state, 0, 6);
    rlt::set(observation, 0,  0 + 0, rlt::get(state, 0, 0));
    rlt::set(observation, 0,  0 + 1, rlt::get(state, 0, 1));
    rlt::set(observation, 0,  0 + 2, rlt::get(state, 0, 2));
    rlt::set(observation, 0,  3 + 0, (1 - 2*qy*qy - 2*qz*qz));
    rlt::set(observation, 0,  3 + 1, (    2*qx*qy - 2*qw*qz));
    rlt::set(observation, 0,  3 + 2, (    2*qx*qz + 2*qw*qy));
    rlt::set(observation, 0,  3 + 3, (    2*qx*qy + 2*qw*qz));
    rlt::set(observation, 0,  3 + 4, (1 - 2*qx*qx - 2*qz*qz));
    rlt::set(observation, 0,  3 + 5, (    2*qy*qz - 2*qw*qx));
    rlt::set(observation, 0,  3 + 6, (    2*qx*qz - 2*qw*qy));
    rlt::set(observation, 0,  3 + 7, (    2*qy*qz + 2*qw*qx));
    rlt::set(observation, 0,  3 + 8, (1 - 2*qx*qx - 2*qy*qy));
    rlt::set(observation, 0, 12 + 0, rlt::get(state, 0, 3 + 4 + 0));
    rlt::set(observation, 0, 12 + 1, rlt::get(state, 0, 3 + 4 + 1));
    rlt::set(observation, 0, 12 + 2, rlt::get(state, 0, 3 + 4 + 2));
    rlt::set(observation, 0, 15 + 0, rlt::get(state, 0, 3 + 4 + 3 + 0));
    rlt::set(observation, 0, 15 + 1, rlt::get(state, 0, 3 + 4 + 3 + 1));
    rlt::set(observation, 0, 15 + 2, rlt::get(state, 0, 3 + 4 + 3 + 2));
}

// Main functions (possibly with side effects)
void rl_tools_init(){
    rlt::malloc(device, buffers);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
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
    return (char*)rlt::checkpoint::meta::name;
}

float rl_tools_test(float* output_mem){
#ifndef RL_TOOLS_DISABLE_TEST
    rlt::evaluate(device, rlt::checkpoint::actor::model, rlt::checkpoint::observation::container, output, buffers);
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(rlt::get(output, 0, i) - rlt::get(rlt::checkpoint::action::container, 0, i));
        output_mem[i] = rlt::get(rlt::checkpoint::action::container, 0, i);
    }
    return acc;
#else
    return 0;
#endif
}

void rl_tools_control(float* state, float* actions){
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 13, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(T*)state}; 
    auto state_rotation_matrix_input = rlt::view(device, input, rlt::matrix::ViewSpec<1, 18>{}, 0, 0);
    observe_rotation_matrix(state_matrix, state_rotation_matrix_input);
#ifdef RL_TOOLS_ACTION_HISTORY
    auto action_history_observation = rlt::view(device, input, rlt::matrix::ViewSpec<1, ACTION_HISTORY_LENGTH * ACTOR_TYPE::SPEC::OUTPUT_DIM>{}, 0, 18);
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
            rlt::set(action_history_observation, 0, step_i * ACTOR_TYPE::SPEC::OUTPUT_DIM + action_i, action_history[step_i][action_i]);
        }
    }
#endif
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)actions};
    rlt::evaluate(device, rlt::checkpoint::actor::model, input, output, buffers);
#ifdef RL_TOOLS_ACTION_HISTORY
    int substep = controller_tick % CONTROL_FREQUENCY_MULTIPLE;
    if(substep == 0){
        for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH - 1; step_i++){
            for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
                action_history[step_i][action_i] = action_history[step_i + 1][action_i];
            }
        }
    }
    for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
        T value = action_history[ACTION_HISTORY_LENGTH - 1][action_i];
        value *= substep;
        value += rlt::get(output, 0, action_i);
        value /= substep + 1;
        action_history[ACTION_HISTORY_LENGTH - 1][action_i] = value;
    }
#endif
    controller_tick++;
}
