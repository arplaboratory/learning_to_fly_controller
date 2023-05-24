#include "debug.h"
#include "usec_time.h"
// #include "policy_generated.h"
#include <math.h>
#include "math3d.h"
#include "log.h"
#include "param.h"
#include "motors.h"
#include "watchdog.h"
#include "controller_pid.h"
#include "power_distribution.h"
#include "backprop_tools_adapter.h"
#include "stabilizer.h"

// #include "dynamics_encoder.h"

// #define DEBUG_OUTPUT_INTERVAL 500
#define CONTROL_INTERVAL_MS 10
#define CONTROL_INTERVAL_US (CONTROL_INTERVAL_MS * 1000)
#define POS_DISTANCE_LIMIT 0.3f
#define CONTROL_PACKET_TIMEOUT_USEC (1000*200)
#define BEHIND_SCHEDULE_MESSAGE_MIN_INTERVAL (1000000)
#define CONTROL_INVOCATION_INTERVAL_ALPHA 0.95f
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
static float target_height = 0.2;

// NN input
static float state_input[13];
static float action_output[4];
// static float position[3];
// static float attitude_quat[4];
// static float twist_linear[3];
// static float twist_angular[3];

// Control variables: output
const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};
static uint8_t set_motors_overwrite = 0;
static uint16_t motor_cmd[4];
static float motor_cmd_divider;
static bool prev_set_motors;

static motors_thrust_uncapped_t motorThrustUncapped;
static motors_thrust_uncapped_t motorThrustBatCompUncapped;
static motors_thrust_pwm_t motorPwm;



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

static inline void update_state(const sensorData_t* sensors, const state_t* state){
  state_input[ 0] = clip(state->position.x - target_pos[0], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  state_input[ 1] = clip(state->position.y - target_pos[1], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  state_input[ 2] = clip(state->position.z - target_pos[2], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  state_input[ 3] = state->attitudeQuaternion.w;
  state_input[ 4] = state->attitudeQuaternion.x;
  state_input[ 5] = state->attitudeQuaternion.y;
  state_input[ 6] = state->attitudeQuaternion.z;
  state_input[ 7] = state->velocity.x;
  state_input[ 8] = state->velocity.y;
  state_input[ 9] = state->velocity.z;
  state_input[10] =  radians(sensors->gyro.x);
  state_input[11] = -radians(sensors->gyro.y);
  state_input[12] =  radians(sensors->gyro.z);
}

void learned_controller_packet_received(){
  uint64_t now = usecTimestamp();
  timestamp_last_control_packet_received = now;
}


void controllerOutOfTreeInit(void){
  controller_state = STATE_RESET;
  controller_tick = 0;
  motor_cmd_divider = 15;
  motor_cmd[0] = 0;
  motor_cmd[1] = 0;
  motor_cmd[2] = 0;
  motor_cmd[3] = 0;
  timestamp_last_reset = usecTimestamp();
  prev_set_motors = false;
  timestamp_last_control_packet_received = 0;
  timestamp_last_behind_schedule_message = 0;
  control_invocation_interval = 0;
  forward_tick = 0;
  controllerPidInit();
  backprop_tools_init();
  DEBUG_PRINT("BackpropTools controller: Init\n");
}

bool controllerOutOfTreeTest(void)
{
  float output[4];
  float absdiff = backprop_tools_test(output);
  for(int i = 0; i < 4; i++){
    DEBUG_PRINT("BackpropTools controller: Test output %d: %f\n", i, output[i]);
  }
  DEBUG_PRINT("BackpropTools controller: Test %f\n", absdiff);
  return controllerPidTest();
}

static void batteryCompensation(const motors_thrust_uncapped_t* motorThrustUncapped, motors_thrust_uncapped_t* motorThrustBatCompUncapped)
{
  float supplyVoltage = pmGetBatteryVoltage();

  for (int motor = 0; motor < STABILIZER_NR_OF_MOTORS; motor++)
  {
    motorThrustBatCompUncapped->list[motor] = motorsCompensateBatteryVoltage(motor, motorThrustUncapped->list[motor], supplyVoltage);
  }
}
static void setMotorRatios(const motors_thrust_pwm_t* motorPwm)
{
  motorsSetRatio(MOTOR_M1, motorPwm->motors.m1);
  motorsSetRatio(MOTOR_M2, motorPwm->motors.m2);
  motorsSetRatio(MOTOR_M3, motorPwm->motors.m3);
  motorsSetRatio(MOTOR_M4, motorPwm->motors.m4);
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
}

static inline void every_10000ms(){
  DEBUG_PRINT("control invocation interval %f\n", (double)control_invocation_interval);
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

void controllerOutOfTree(control_t *control, setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  uint64_t now = usecTimestamp();
  watchdogReset();
  control_invocation_interval *= CONTROL_INVOCATION_INTERVAL_ALPHA;
  control_invocation_interval += (1-CONTROL_INVOCATION_INTERVAL_ALPHA) * (now - timestamp_last_control_invocation);
  timestamp_last_control_invocation = now;
  bool set_motors = (now - timestamp_last_control_packet_received < CONTROL_PACKET_TIMEOUT_USEC)  || (set_motors_overwrite == 1 && motor_cmd_divider >= 3);
  if(!prev_set_motors && set_motors){
    target_pos[0] = state->position.x;
    target_pos[1] = state->position.y;
    target_pos[2] = state->position.z + target_height;
    DEBUG_PRINT("Controller activated\n");
  }
  if(prev_set_motors && !set_motors){
    DEBUG_PRINT("Controller deactivated\n");
  }

  trigger_every(tick);
  prev_set_motors = set_motors;


  if (tick % CONTROL_INTERVAL_MS == 0){
    update_state(sensors, state);
    {
      int64_t before = usecTimestamp();
      // backprop_tools_control_rotation_matrix(state_input, action_output);
      int64_t after = usecTimestamp();
      if (tick % (CONTROL_INTERVAL_MS * 1000) == 0){
        DEBUG_PRINT("backprop_tools_run took %lldus\n", after - before);
      }
    }
    for(uint8_t i=0; i<4; i++){
      if (tick % (CONTROL_INTERVAL_MS * 1000) == 0){
        DEBUG_PRINT("action_output[%d]: %f\n", i, action_output[i]);
      }
      float a_pp = (action_output[i] + 1)/2;
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
  if(!set_motors){
    controllerPid(control, setpoint, sensors, state, tick);
    powerDistribution(&control, &motorThrustUncapped);
    batteryCompensation(&motorThrustUncapped, &motorThrustBatCompUncapped);
    powerDistributionCap(&motorThrustBatCompUncapped, &motorPwm);
    setMotorRatios(&motorPwm);
  }
  controller_tick++;
}



PARAM_GROUP_START(pudmrlg)
PARAM_ADD(PARAM_FLOAT, motor_div, &motor_cmd_divider)
PARAM_ADD(PARAM_FLOAT, target_z, &target_height)
PARAM_ADD(PARAM_UINT8, smo, &set_motors_overwrite)
PARAM_GROUP_STOP(pudmrlg)

LOG_GROUP_START(dpudmrlp)
LOG_ADD(LOG_FLOAT, x, &state_input[0])
LOG_ADD(LOG_FLOAT, y, &state_input[1])
LOG_ADD(LOG_FLOAT, z, &state_input[2])
LOG_GROUP_STOP(dpudmrlp)

LOG_GROUP_START(dpudmrlaq)
LOG_ADD(LOG_FLOAT, w, &state_input[3])
LOG_ADD(LOG_FLOAT, x, &state_input[4])
LOG_ADD(LOG_FLOAT, y, &state_input[5])
LOG_ADD(LOG_FLOAT, z, &state_input[6])
LOG_GROUP_STOP(dpudmrlaq)

LOG_GROUP_START(dpudmrltwl)
LOG_ADD(LOG_FLOAT, x, &state_input[7])
LOG_ADD(LOG_FLOAT, y, &state_input[8])
LOG_ADD(LOG_FLOAT, z, &state_input[9])
LOG_GROUP_STOP(dpudmrltwl)

LOG_GROUP_START(dpudmrltwa)
LOG_ADD(LOG_FLOAT, x, &state_input[10])
LOG_ADD(LOG_FLOAT, y, &state_input[11])
LOG_ADD(LOG_FLOAT, z, &state_input[12])
LOG_GROUP_STOP(dpudmrltwa)


LOG_GROUP_START(dpudmrlm)
LOG_ADD(LOG_UINT16, m1, &motor_cmd[0])
LOG_ADD(LOG_UINT16, m2, &motor_cmd[1])
LOG_ADD(LOG_UINT16, m3, &motor_cmd[2])
LOG_ADD(LOG_UINT16, m4, &motor_cmd[3])
LOG_GROUP_STOP(dpudmrlm)

