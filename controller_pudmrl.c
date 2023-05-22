#include "controller_pudmrl.h"
#include "debug.h"
#include "usec_time.h"
#include "policy_generated.h"
#include <math.h>
#include "math3d.h"
#include "log.h"
#include "param.h"
#include "motors.h"
#include "watchdog.h"
#include "controller_pid.h"
#include "power_distribution.h"

#include "dynamics_encoder.h"

// #define DEBUG_OUTPUT_INTERVAL 500
#define CONTROL_INTERVAL_MS 2
#define CONTROL_INTERVAL_US (CONTROL_INTERVAL_MS * 1000)
#define CONTROL_INTERVAL_EXECUTION_TO_TRAINING_FACTOR ((int)((1/(float)NN_POLICY_TRAINING_CONTROL_FREQUENCY)/((float)CONTROL_INTERVAL_MS/1000) + 0.5f))
#define FORWARD_STEPS_PER_ITERATION 1000
#define POS_DISTANCE_LIMIT 0.3f
#define CONTROL_PACKET_TIMEOUT_USEC (1000*200)
#define BEHIND_SCHEDULE_MESSAGE_MIN_INTERVAL (1000000)
#define CONTROL_INVOCATION_INTERVAL_ALPHA 0.95f
// #define NN_TEST
// #define DEBUG_INPUT_OUTPUT
#define DEBUG_MEASURE_FORWARD_TIME

// #define PRINT_RPY

// test stuff

typedef enum ControllerState{
  STATE_RESET,
  STATE_FORWARD,
  STATE_FINISHED 
} ControllerState;

// State
static ControllerState controller_state;

// Counters
static uint64_t controller_tick = 0; // Number of control function invocations
static uint64_t forward_tick = 0; // Number of forward passes

// Timestamps
static uint64_t timestamp_last_reset;
static uint64_t timestamp_last_behind_schedule_message = 0;
static uint64_t timestamp_last_control_invocation = 0;
static uint64_t timestamp_last_control_packet_received = 0;

// Logging variables
static float control_invocation_interval = 0;

// Control variables: input
static float target_pos[3] = {0, 0, 0}; // described in global enu frame
static float target_height = 0.0;

// NN input
static float position[3];
static float attitude_quat[4];
static float attitude_mat[9];
static float attitude_rpy[3];
static float twist_linear[3];
static float twist_angular[3];
static float action_history_accumulator[NN_OUTPUT_DIM];
static float action_history[NN_POLICY_ACTION_HISTORY][NN_OUTPUT_DIM];

// Counter
static uint16_t action_history_pos = 0;


// Control variables: output
const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};
static uint8_t set_motors_overwrite = 0;
static uint16_t motor_cmd[4];
static float motor_cmd_divider;
static bool prev_set_motors;



void controller_pudmrl_control_packet_received(){
  uint64_t now = usecTimestamp();
  timestamp_last_control_packet_received = now;
}

static inline float clip(float v, float low, float high){
  if(v < low){
    return low;
  }
  else{
    if(v > high){
      return high;
    }
    else{
      return v;
    }
  }
}

// static inline void convert_quaternion_to_rpy(const quaternion_t* q, float v[3]) {
// 	// from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
// 	float x = q->x;
// 	float y = q->y;
// 	float z = q->z;
// 	float w = q->w;
// 	v[1] = atan2f(2.0f * (w * x + y * z), 1 - 2 * (fsqr(x) + fsqr(y))); // roll
// 	v[0] = -asinf(2.0f * (w * y - x * z)); // pitch
// 	v[2] = atan2f(2.0f * (w * z + x * y), 1 - 2 * (fsqr(y) + fsqr(z))); // yaw
// }

static inline void convert_rotation_matrix_to_rpy_extrinsic(float* m, float v[3]) {
  v[2] = atan2f( m[3*1 + 0],m[3*0 + 0]);
  v[1] = atan2f(-m[3*2 + 0],sqrtf(m[3*2 + 1]*m[3*2 + 1]+m[3*2 + 2]*m[3*2 + 2]));
  v[0] = atan2f( m[3*2 + 1],m[3*2 + 2]);
}

static inline void convert_quaternion_to_rotation_matrix(const quaternion_t* q, float* m) {
	// from https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
	float x = q->x;
	float y = q->y;
	float z = q->z;
	float w = q->w;

  // // Crazy
	// m[0][0] = 1 - 2*y*y - 2*z*z;
	// m[0][1] = 2*x*y - 2*z*w;
	// m[0][2] = 2*x*z + 2*y*w,
	// m[1][0] = 2*x*y + 2*z*w;
	// m[1][1] = 1 - 2*x*x - 2*z*z;
	// m[1][2] = 2*y*z - 2*x*w,
	// m[2][0] = 2*x*z - 2*y*w;
	// m[2][1] = 2*y*z + 2*x*w;
	// m[2][2] = 1 - 2*x*x - 2*y*y;

  // // NED
	// m[0][0] = 1 - 2*y*y - 2*z*z;
	// m[0][1] = 2*x*y - 2*z*w;
	// m[0][2] = 2*x*z + 2*y*w,
	// m[1][0] = - (2*x*y + 2*z*w);
	// m[1][1] = - (1 - 2*x*x - 2*z*z);
	// m[1][2] = - (2*y*z - 2*x*w),
	// m[2][0] = - (2*x*z - 2*y*w);
	// m[2][1] = - (2*y*z + 2*x*w);
	// m[2][2] = - (1 - 2*x*x - 2*y*y);

  // ENU
  //       0  -1  0
  // 1R2 = 1   0  0
  //       0   0  1
  // 2R3 = IMU 
  // 1R3 = 1R2 * 2R3
  // wanted: diff between 1R3 and 1R2 in 1
  // 1R3 = X * 1R2
  // X = 1R3 * 1R2'
  // X = 1R2 * 2R3 * 1R2'
  // X = 1R2 * IMU * 1R2'
  //     1  2  3
  // A = 4  5  6 =! IMU
  //     7  8  9
  // X = 1R2 * A * 1R2'
  //      5  -4  -6
  // X = -2   1   3
  //     -8   7   9
    
    
    

	m[1*3 + 1] =  (1 - 2*y*y - 2*z*z); // 1
	m[1*3 + 0] = -(2*x*y - 2*z*w);     // 2
	m[1*3 + 2] =  (2*x*z + 2*y*w);     // 3
	m[0*3 + 1] = -(2*x*y + 2*z*w);     // 4
	m[0*3 + 0] =  (1 - 2*x*x - 2*z*z); // 5
	m[0*3 + 2] = -(2*y*z - 2*x*w);     // 6
	m[2*3 + 1] =  (2*x*z - 2*y*w);     // 7
	m[2*3 + 0] = -(2*y*z + 2*x*w);     // 8
	m[2*3 + 2] =  (1 - 2*x*x - 2*y*y); // 9
}


static inline void write_action_history(unsigned int offset){
  #if NN_POLICY_ACTION_HISTORY > 0
    for(uint16_t i=0; i < NN_POLICY_ACTION_HISTORY; i++){
      for(uint16_t j=0; j < NN_OUTPUT_DIM; j++){
        result_double_buffers[0][offset + i*NN_OUTPUT_DIM_T + j] = action_history[(action_history_pos + 1 + i) % NN_POLICY_ACTION_HISTORY][j]; // action_history_pos + 1 because action_history_pos is the latest one and +1 is the oldest one
      }
    }
  #endif
}


void update_controller_input_ang_vel(){
  result_double_buffers[0][0] = twist_angular[0];
  result_double_buffers[0][1] = twist_angular[1];
  result_double_buffers[0][2] = twist_angular[2];
}

void update_controller_input_att_ang_vel(){
  result_double_buffers[0][0 + 0]  = attitude_mat[0 * 3 + 0];
  result_double_buffers[0][0 + 1]  = attitude_mat[1 * 3 + 0];
  result_double_buffers[0][0 + 2]  = attitude_mat[2 * 3 + 0];
  result_double_buffers[0][0 + 3]  = attitude_mat[0 * 3 + 1];
  result_double_buffers[0][0 + 4]  = attitude_mat[1 * 3 + 1];
  result_double_buffers[0][0 + 5]  = attitude_mat[2 * 3 + 1];
  result_double_buffers[0][0 + 6]  = attitude_mat[0 * 3 + 2];
  result_double_buffers[0][0 + 7]  = attitude_mat[1 * 3 + 2];
  result_double_buffers[0][0 + 8]  = attitude_mat[2 * 3 + 2];
  result_double_buffers[0][9 + 0]  = attitude_rpy[0];
  result_double_buffers[0][9 + 1]  = attitude_rpy[1];
  result_double_buffers[0][9 + 2]  = attitude_rpy[2];
  result_double_buffers[0][12 + 0] = twist_angular[0];
  result_double_buffers[0][12 + 1] = twist_angular[1];
  result_double_buffers[0][12 + 2] = twist_angular[2];
}

void update_controller_input_vel_att_ang_vel(){
  result_double_buffers[0][0 + 0]  = attitude_mat[0 * 3 + 0];
  result_double_buffers[0][0 + 1]  = attitude_mat[1 * 3 + 0];
  result_double_buffers[0][0 + 2]  = attitude_mat[2 * 3 + 0];
  result_double_buffers[0][0 + 3]  = attitude_mat[0 * 3 + 1];
  result_double_buffers[0][0 + 4]  = attitude_mat[1 * 3 + 1];
  result_double_buffers[0][0 + 5]  = attitude_mat[2 * 3 + 1];
  result_double_buffers[0][0 + 6]  = attitude_mat[0 * 3 + 2];
  result_double_buffers[0][0 + 7]  = attitude_mat[1 * 3 + 2];
  result_double_buffers[0][0 + 8]  = attitude_mat[2 * 3 + 2];
  result_double_buffers[0][9 + 0]  = attitude_rpy[0];
  result_double_buffers[0][9 + 1]  = attitude_rpy[1];
  result_double_buffers[0][9 + 2]  = attitude_rpy[2];
  result_double_buffers[0][12 + 0] = twist_linear[0];
  result_double_buffers[0][12 + 1] = twist_linear[1];
  result_double_buffers[0][12 + 2] = twist_linear[2];
  result_double_buffers[0][15 + 0] = twist_angular[0];
  result_double_buffers[0][15 + 1] = twist_angular[1];
  result_double_buffers[0][15 + 2] = twist_angular[2];
}

void update_controller_input_pos_vel_att_ang_vel(){
  result_double_buffers[0][0 + 0]  = position[0];
  result_double_buffers[0][0 + 1]  = position[1];
  result_double_buffers[0][0 + 2]  = position[2];
  result_double_buffers[0][3 + 0]  = attitude_mat[0 * 3 + 0];
  result_double_buffers[0][3 + 1]  = attitude_mat[1 * 3 + 0];
  result_double_buffers[0][3 + 2]  = attitude_mat[2 * 3 + 0];
  result_double_buffers[0][3 + 3]  = attitude_mat[0 * 3 + 1];
  result_double_buffers[0][3 + 4]  = attitude_mat[1 * 3 + 1];
  result_double_buffers[0][3 + 5]  = attitude_mat[2 * 3 + 1];
  result_double_buffers[0][3 + 6]  = attitude_mat[0 * 3 + 2];
  result_double_buffers[0][3 + 7]  = attitude_mat[1 * 3 + 2];
  result_double_buffers[0][3 + 8]  = attitude_mat[2 * 3 + 2];
  result_double_buffers[0][12 + 0] = attitude_rpy[0];
  result_double_buffers[0][12 + 1] = attitude_rpy[1];
  result_double_buffers[0][12 + 2] = attitude_rpy[2];
  result_double_buffers[0][15 + 0] = twist_linear[0];
  result_double_buffers[0][15 + 1] = twist_linear[1];
  result_double_buffers[0][15 + 2] = twist_linear[2];
  result_double_buffers[0][18 + 0] = twist_angular[0];
  result_double_buffers[0][18 + 1] = twist_angular[1];
  result_double_buffers[0][18 + 2] = twist_angular[2];
}

void update_controller_input_pos_vel_att_ang_vel_action_history(){
  result_double_buffers[0][0 + 0]  = position[0];
  result_double_buffers[0][0 + 1]  = position[1];
  result_double_buffers[0][0 + 2]  = position[2];
  result_double_buffers[0][3 + 0]  = attitude_mat[0 * 3 + 0];
  result_double_buffers[0][3 + 1]  = attitude_mat[1 * 3 + 0];
  result_double_buffers[0][3 + 2]  = attitude_mat[2 * 3 + 0];
  result_double_buffers[0][3 + 3]  = attitude_mat[0 * 3 + 1];
  result_double_buffers[0][3 + 4]  = attitude_mat[1 * 3 + 1];
  result_double_buffers[0][3 + 5]  = attitude_mat[2 * 3 + 1];
  result_double_buffers[0][3 + 6]  = attitude_mat[0 * 3 + 2];
  result_double_buffers[0][3 + 7]  = attitude_mat[1 * 3 + 2];
  result_double_buffers[0][3 + 8]  = attitude_mat[2 * 3 + 2];
  result_double_buffers[0][12 + 0] = attitude_rpy[0];
  result_double_buffers[0][12 + 1] = attitude_rpy[1];
  result_double_buffers[0][12 + 2] = attitude_rpy[2];
  result_double_buffers[0][15 + 0] = twist_linear[0];
  result_double_buffers[0][15 + 1] = twist_linear[1];
  result_double_buffers[0][15 + 2] = twist_linear[2];
  result_double_buffers[0][18 + 0] = twist_angular[0];
  result_double_buffers[0][18 + 1] = twist_angular[1];
  result_double_buffers[0][18 + 2] = twist_angular[2];
  write_action_history(21);
}

static inline void update_controller_input(const sensorData_t* sensors, const state_t* state){
  position[1] = clip(state->position.x - target_pos[1], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT); // global frame
  position[0] = clip(-state->position.y - target_pos[0], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  position[2] = clip(state->position.z - target_pos[2], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);

  convert_quaternion_to_rotation_matrix(&state->attitudeQuaternion, attitude_mat); // global frame
  convert_rotation_matrix_to_rpy_extrinsic(attitude_mat, attitude_rpy); // global frame

  attitude_quat[0] = state->attitudeQuaternion.w;
  attitude_quat[1] = state->attitudeQuaternion.x;
  attitude_quat[2] = state->attitudeQuaternion.y;
  attitude_quat[3] = state->attitudeQuaternion.z;

  twist_linear[1] = state->velocity.x; // global frame
  twist_linear[0] = -state->velocity.y;
  twist_linear[2] = state->velocity.z;
  twist_angular[0] = radians(sensors->gyro.x); // local frame
  twist_angular[1] = -radians(sensors->gyro.y);
  twist_angular[2] = -radians(sensors->gyro.z);

  switch(NN_INPUT_DIM){
    case 3:
      update_controller_input_ang_vel();
    break;
    case 15:
      update_controller_input_att_ang_vel();
    break;
    case 18:
      update_controller_input_vel_att_ang_vel();
    break;
    case 21:
      update_controller_input_pos_vel_att_ang_vel();
    break;
    default:

      update_controller_input_pos_vel_att_ang_vel_action_history();
      
    break;
  }
}




void controllerPudmrlInit(void)
{
  controller_state = STATE_RESET;
  controller_tick = 0;
  motor_cmd_divider = 15;
  motor_cmd[0] = 0;
  motor_cmd[1] = 0;
  motor_cmd[2] = 0;
  motor_cmd[3] = 0;
  timestamp_last_reset = usecTimestamp();
  prev_set_motors = false;
  for(int i=0; i < NN_POLICY_ACTION_HISTORY; i++){
    for(int j=0; j < NN_OUTPUT_DIM; j++){
      if (i == 0){
        action_history_accumulator[j] = 0;
      }
      action_history[i][j] = 0;
    }
  }
  action_history_pos = 0;
  timestamp_last_control_packet_received = 0;
  timestamp_last_behind_schedule_message = 0;
  control_invocation_interval = 0;
  forward_tick = 0;
  controllerPidInit();
  DEBUG_PRINT("PUDM-RL: Init\n");
}

bool controllerPudmrlTest(void)
{
  set_forward_state(0);
  // for(uint16_t i=0; i < NN_INPUT_DIM_T; i++){
  //   result_double_buffers[0][i] = 0.1;
  // }
  // forward();
  // for(uint16_t i=0; i < NN_OUTPUT_DIM_T; i++){
  //   printf("Test output %i: %f\n", i, result_double_buffers[NN_OUTPUT_TOCK ? 0 : 1][i] = 0.1);
  // }
  // set_forward_state(0);
  controllerPidTest();

  return true;
}




static inline void every_500ms(){
#ifdef PRINT_TWIST
  DEBUG_PRINT("tw.l: %5.2f, %5.2f, %5.2f tw.a: %5.2f, %5.2f, %5.2f\n", twist_linear[0], twist_linear[1], twist_linear[2], twist_angular[0], twist_angular[1], twist_angular[2]);
#endif
}

static inline void every_1000ms(){
#ifdef PRINT_RPY
  DEBUG_PRINT("rpy: %5.2f, %5.2f, %5.2f\n", attitude_rpy[0], attitude_rpy[1], attitude_rpy[2]);
#endif
  // #ifdef NN_ARM_OPTIMIZATIONS
  // uint64_t before = usecTimestamp();
  // reset_recurrent_arm();
  // forward_recurrent_arm(test_data, test_data_seq_len);
  // // forward_recurrent_arm_step(test_data);
  // uint64_t after = usecTimestamp();
  // DEBUG_PRINT("Recurrent forward took %lld uSec\n", after-before);
  // for(int i=0; i < recurrent_output_dim && i < 5; i ++){
  //     DEBUG_PRINT("%f\n", (double)recurrent_output_arm[i]);
  // }
  // #else
  // uint64_t before = usecTimestamp();
  // reset_recurrent();
  // forward_recurrent(test_data, test_data_seq_len);
  // uint64_t after = usecTimestamp();
  // DEBUG_PRINT("Recurrent forward took %lld uSec\n", after-before);
  // for(int i=0; i < recurrent_output_dim && i < 5; i ++){
  //     DEBUG_PRINT("%f\n", (double)recurrent_output(test_data_seq_len)[i]);
  // }
  // #endif

}

static inline void every_10000ms(){
  DEBUG_PRINT("control invocation interval %f\n", (double)control_invocation_interval);
  DEBUG_PRINT("Control factor %d", CONTROL_INTERVAL_EXECUTION_TO_TRAINING_FACTOR);
}

static inline void trigger_every(uint64_t controller_tick){
  if(controller_tick > 3000){
    if(controller_tick % 500 == 0){
      every_500ms();
    }
    if(controller_tick  % 1000 == 150){
      every_1000ms();
    }
    if(controller_tick % 10000 == 9300){
      every_10000ms();
    }
  }
}

#ifndef NN_TEST
void controllerPudmrl(control_t *control, setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
{
  uint64_t now = usecTimestamp();
  watchdogReset();
  control_invocation_interval *= CONTROL_INVOCATION_INTERVAL_ALPHA;
  control_invocation_interval += (1-CONTROL_INVOCATION_INTERVAL_ALPHA) * (now - timestamp_last_control_invocation);
  timestamp_last_control_invocation = now;
  bool set_motors = (now - timestamp_last_control_packet_received < CONTROL_PACKET_TIMEOUT_USEC)  || (set_motors_overwrite == 1 && motor_cmd_divider >= 3);
  if(!prev_set_motors && set_motors){
    target_pos[1] = state->position.x;
    target_pos[0] = -state->position.y;
    target_pos[2] = state->position.z + target_height;
  }
  trigger_every(tick);
  prev_set_motors = set_motors;


  if (tick % CONTROL_INTERVAL_MS == 0){
    #ifdef DEBUG_MEASURE_FORWARD_TIME
    uint64_t before = usecTimestamp();
    #endif
    update_controller_input(sensors, state);
    #ifdef NN_ARM_OPTIMIZATIONS
    arm_forward();
    #else
    // forward();
    #endif
    #ifdef DEBUG_MEASURE_FORWARD_TIME
    uint64_t after = usecTimestamp();
    if(tick % 2300 == 0){
      DEBUG_PRINT("Forward (incl. controller update) took %lld\n", after-before);
    }
    #endif
    bool action_history_update = tick % (CONTROL_INTERVAL_MS * CONTROL_INTERVAL_EXECUTION_TO_TRAINING_FACTOR) == 0;
    if (action_history_update){
      #if NN_POLICY_ACTION_HISTORY > 0
        action_history_pos = (action_history_pos + 1) % NN_POLICY_ACTION_HISTORY;
      #endif
    }
    for(uint8_t i=0; i<4; i++){
      float a_pp = (result_double_buffers[NN_OUTPUT_TOCK ? 0 : 1][i] + 1)/2;
      action_history_accumulator[i] += a_pp;

      if(action_history_update){
        #if NN_POLICY_ACTION_HISTORY > 0
          action_history[action_history_pos][i] = action_history_accumulator[i]/NN_POLICY_ACTION_HISTORY;
          action_history_accumulator[i] = 0;
        #endif
      }
      motor_cmd[i] = a_pp * UINT16_MAX;
      if(set_motors){
        motorsSetRatio(motors[i], clip((float)motor_cmd[i] / motor_cmd_divider, 0, UINT16_MAX));
      }
    }
    int64_t spare_time = CONTROL_INTERVAL_US - (now - timestamp_last_reset) ;
    if(spare_time < 0 && (now - timestamp_last_behind_schedule_message > BEHIND_SCHEDULE_MESSAGE_MIN_INTERVAL)){
      DEBUG_PRINT("PUDM-RL Controller is behind schedule: %lldus/%dus\n", (int64_t)(now-timestamp_last_reset), CONTROL_INTERVAL_US);
      timestamp_last_behind_schedule_message = now;
    }
    timestamp_last_reset = usecTimestamp();
  }
  else{
    // forward_recurrent_arm_step(test_data);
  }
  if(!set_motors){
    controllerPid(control, setpoint, sensors, state, tick);
    powerDistribution(control);
  }
  controller_tick++;
}
#endif
#ifdef NN_TEST
#include "test_states.h"
#define NN_TEST_COUNT 1
#define TARGET_STATE 1
void controllerPudmrl(control_t *control, setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
{
  if(tick % 1000 == 0){
    set_forward_state(0);
    for(uint16_t i=0; i < NN_INPUT_DIM_T; i++){
      result_double_buffers[0][i] = test_states[NN_INPUT_DIM * TARGET_STATE + i];
    }
    arm_forward();
    for(uint16_t i=0; i < NN_OUTPUT_DIM_T; i++){
      DEBUG_PRINT("Test output %i: %f\n", i, result_double_buffers[NN_OUTPUT_TOCK ? 0 : 1][i]*1000);
    }
    set_forward_state(0);
  }
  if(tick % 1000 == 500){
    uint64_t before = usecTimestamp();
    for(int i=0; i < NN_TEST_COUNT; i++){
        bool finished = true;
        arm_forward();
        // while(!finished){
        //   finished = forward_step_by_step(FORWARD_STEPS_PER_ITERATION);
        // }
    }
    uint64_t after = usecTimestamp();
    DEBUG_PRINT("Forward took %lld\n", after-before);
  }

}
#endif

PARAM_GROUP_START(pudmrlg)
PARAM_ADD(PARAM_FLOAT, motor_div, &motor_cmd_divider)
PARAM_ADD(PARAM_FLOAT, target_z, &target_height)
PARAM_ADD(PARAM_UINT8, smo, &set_motors_overwrite)
PARAM_GROUP_STOP(pudmrlg)

LOG_GROUP_START(dpudmrlp)
LOG_ADD(LOG_FLOAT, x, &position[0])
LOG_ADD(LOG_FLOAT, y, &position[1])
LOG_ADD(LOG_FLOAT, z, &position[2])
LOG_GROUP_STOP(dpudmrlp)

LOG_GROUP_START(dpudmrlarpy)
LOG_ADD(LOG_FLOAT, roll,  &attitude_rpy[0])
LOG_ADD(LOG_FLOAT, pitch, &attitude_rpy[1])
LOG_ADD(LOG_FLOAT, yaw,   &attitude_rpy[2])
LOG_GROUP_STOP(dpudmrlarpy)

LOG_GROUP_START(dpudmrlaq)
LOG_ADD(LOG_FLOAT, w, &attitude_quat[0])
LOG_ADD(LOG_FLOAT, x, &attitude_quat[1])
LOG_ADD(LOG_FLOAT, y, &attitude_quat[2])
LOG_ADD(LOG_FLOAT, z, &attitude_quat[3])
LOG_GROUP_STOP(dpudmrlaq)

LOG_GROUP_START(dpudmrltwl)
LOG_ADD(LOG_FLOAT, x, &twist_linear[0])
LOG_ADD(LOG_FLOAT, y, &twist_linear[1])
LOG_ADD(LOG_FLOAT, z, &twist_linear[2])
LOG_GROUP_STOP(dpudmrltwl)

LOG_GROUP_START(dpudmrltwa)
LOG_ADD(LOG_FLOAT, x, &twist_angular[0])
LOG_ADD(LOG_FLOAT, y, &twist_angular[1])
LOG_ADD(LOG_FLOAT, z, &twist_angular[2])
LOG_GROUP_STOP(dpudmrltwa)


LOG_GROUP_START(dpudmrlm)
LOG_ADD(LOG_UINT16, m1, &motor_cmd[0])
LOG_ADD(LOG_UINT16, m2, &motor_cmd[1])
LOG_ADD(LOG_UINT16, m3, &motor_cmd[2])
LOG_ADD(LOG_UINT16, m4, &motor_cmd[3])
LOG_GROUP_STOP(dpudmrlm)

