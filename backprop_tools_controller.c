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
#include "controller_mellinger.h"
#include "controller_indi.h"
#include "controller_brescianini.h"
#include "power_distribution.h"
#include "backprop_tools_adapter.h"
#include "stabilizer_types.h"
#include "pm.h"
#include "task.h"

// #include "dynamics_encoder.h"

// #define DEBUG_OUTPUT_INTERVAL 500
#define CONTROL_INTERVAL_MS 2
#define CONTROL_INTERVAL_US (CONTROL_INTERVAL_MS * 1000)
#define CONTROL_PACKET_TIMEOUT_USEC (1000*400)
#define BEHIND_SCHEDULE_MESSAGE_MIN_INTERVAL (1000000)
#define CONTROL_INVOCATION_INTERVAL_ALPHA 0.95f
#define DEBUG_MEASURE_FORWARD_TIME
// #define MIN_RPM 10000
#define MIN_RPM 0
#define MAX_RPM 21702.1
// #define WAYPOINT_NAVIGATION
// static uint8_t waypoint_navigation = 0;
#define WAYPOINT_NAVIGATION_POINT_DURATION (4 * 1000 * 1000)
#define WAYPOINT_NAVIGATION_POINTS (5)

// #define PRINT_RPY
// #define PRINT_TWIST

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
static float target_pos[3] = {0, 0, 0};
static float target_vel[3] = {0, 0, 0};
static float pos_error[3] = {0, 0, 0};
static float relative_pos[3] = {0, 0, 0};
static float origin[3] = {0, 0, 0};

static float POS_DISTANCE_LIMIT;
static float VEL_DISTANCE_LIMIT;
static float POS_DISTANCE_LIMIT_MELLINGER;
static float VEL_DISTANCE_LIMIT_MELLINGER;
static float POS_DISTANCE_LIMIT_BRESCIANI;
static float VEL_DISTANCE_LIMIT_BRESCIANI;
static uint8_t MELLINGER_ENABLE_INTEGRATORS;

enum Mode{
  POSITION = 0,
  WAYPOINT_NAVIGATION = 1,
  WAYPOINT_NAVIGATION_DYNAMIC = 2,
  FIGURE_EIGHT = 3
};
static uint8_t mode;
static float trajectory[WAYPOINT_NAVIGATION_POINTS][3] = {
  {0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0},
  {1.0, 1.0, 0.0},
  {0.0, 1.0, 0.0},
  {0.0, 0.0, 0.0},
};
static uint8_t waypoint_navigation_dynamic_current_waypoint = 0;
static float waypoint_navigation_dynamic_threshold = 0.1;
static float figure_eight_interval = 5.5;
static float figure_eight_warmup_time; 
static float figure_eight_scale = 1.0;
static float trajectory_scale = 0.5;
static float figure_eight_progress = 0;
static uint64_t figure_eight_last_invocation;
static float target_height;
static uint64_t timestamp_last_waypoint, timestamp_controller_activation;
static uint8_t log_set_motors = 0;

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

static setpoint_t last_setpoint;

static uint8_t hand_test = 0; // 0 = off; 1 = setpoint; 2 = angular velocity rejection; 3 = angular velocity rejection + orientation rejection;
static uint8_t use_orig_controller = 0;



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
  if(hand_test == 0){
    state_input[ 0] = clip(state->position.x - target_pos[0], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
    state_input[ 1] = clip(state->position.y - target_pos[1], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
    state_input[ 2] = clip(state->position.z - target_pos[2], -POS_DISTANCE_LIMIT, POS_DISTANCE_LIMIT);
  }
  else{
    state_input[ 0] = 0;
    state_input[ 1] = 0;
    state_input[ 2] = 0;
  }
  if(hand_test == 0 || hand_test == 3){
    state_input[ 3] = state->attitudeQuaternion.w;
    state_input[ 4] = state->attitudeQuaternion.x;
    state_input[ 5] = state->attitudeQuaternion.y;
    state_input[ 6] = state->attitudeQuaternion.z;
  }
  else{
    state_input[ 3] = 1;
    state_input[ 4] = 0;
    state_input[ 5] = 0;
    state_input[ 6] = 0;
  }
  if(hand_test == 0){
    state_input[ 7] = clip(state->velocity.x - target_vel[0], -VEL_DISTANCE_LIMIT, VEL_DISTANCE_LIMIT);
    state_input[ 8] = clip(state->velocity.y - target_vel[1], -VEL_DISTANCE_LIMIT, VEL_DISTANCE_LIMIT);
    state_input[ 9] = clip(state->velocity.z - target_vel[2], -VEL_DISTANCE_LIMIT, VEL_DISTANCE_LIMIT);
  }
  else{
    state_input[ 7] = 0;
    state_input[ 8] = 0;
    state_input[ 9] = 0;
  }
  if(hand_test != 1){
    state_input[10] = radians(sensors->gyro.x);
    state_input[11] = radians(sensors->gyro.y);
    state_input[12] = radians(sensors->gyro.z);
  }
  else{
    state_input[10] = 0;
    state_input[11] = 0;
    state_input[12] = 0;
  }
}

void learned_controller_packet_received(){
  uint64_t now = usecTimestamp();
  timestamp_last_control_packet_received = now;
}


void controllerOutOfTreeInit(void){
  controller_state = STATE_RESET;
  controller_tick = 0;
  motor_cmd_divider = 1.0;
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
  hand_test = 0;
  timestamp_last_waypoint = 0;
  trajectory_scale = 0.5;
  relative_pos[0] = 0;
  relative_pos[1] = 0;
  relative_pos[2] = 0;
  log_set_motors = 0;

  POS_DISTANCE_LIMIT = 0.2f;
  VEL_DISTANCE_LIMIT = 1.0f;
  POS_DISTANCE_LIMIT_MELLINGER = 0.2f;
  VEL_DISTANCE_LIMIT_MELLINGER = 1.0f;
  POS_DISTANCE_LIMIT_BRESCIANI = 0.2f;
  VEL_DISTANCE_LIMIT_BRESCIANI = 1.0f;
  MELLINGER_ENABLE_INTEGRATORS = 1;

  target_height = 0.0;

  // mode = POSITION;
  mode = FIGURE_EIGHT;
  use_orig_controller = 0;
  waypoint_navigation_dynamic_current_waypoint = 0;
  waypoint_navigation_dynamic_threshold = 0;

  figure_eight_interval = 5.5;
  figure_eight_scale = 1;
  figure_eight_progress = 0;
  figure_eight_warmup_time = 2;

  controllerPidInit();
  controllerMellingerFirmwareInit();
  controllerINDIInit();
  controllerBrescianiniInit();
  backprop_tools_init();

  DEBUG_PRINT("BackpropTools controller init! Checkpoint: %s\n", backprop_tools_get_checkpoint_name());
}

bool controllerOutOfTreeTest(void)
{
  float output[4];
  float absdiff = backprop_tools_test(output);
  if(absdiff < 0){
    absdiff = -absdiff;
  }
  DEBUG_PRINT("BackpropTools controller test, abs diff: %f\n", absdiff);
  for(int i = 0; i < 4; i++){
    DEBUG_PRINT("BackpropTools controller: Test action %d: %f\n", i, output[i]);
  }
  if(absdiff > 0.2){
    return false;
  }
  return controllerPidTest() && controllerMellingerFirmwareTest() && controllerINDITest() && controllerBrescianiniTest();
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
  DEBUG_PRINT("tw.l: %5.2f, %5.2f, %5.2f tw.a: %5.2f, %5.2f, %5.2f\n", state_input[7], state_input[8], state_input[9], state_input[10], state_input[11], state_input[12]);
  DEBUG_PRINT("q: %5.2f, %5.2f, %5.2f, %5.2f\n", state_input[3], state_input[4], state_input[5], state_input[6]);
#endif
}

static inline void every_1000ms(){
#ifdef PRINT_RPY
  DEBUG_PRINT("rpy: %5.2f, %5.2f, %5.2f\n", attitude_rpy[0], attitude_rpy[1], attitude_rpy[2]);
#endif

  DEBUG_PRINT("Last setpoint: x disposition/mode %f/%d\n", last_setpoint.position.x, last_setpoint.mode.x);
  DEBUG_PRINT("Last setpoint: y disposition/mode %f/%d\n", last_setpoint.position.y, last_setpoint.mode.y);
  DEBUG_PRINT("Last setpoint: z disposition/mode %f/%d\n", last_setpoint.position.z, last_setpoint.mode.z);
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
  if(setpoint->mode.x == modeVelocity && setpoint->mode.y == modeVelocity && setpoint->mode.z == modeAbs){
    // timestamp_last_control_packet_received = now;
  }

  last_setpoint = *setpoint;
  watchdogReset();
  control_invocation_interval *= CONTROL_INVOCATION_INTERVAL_ALPHA;
  control_invocation_interval += (1-CONTROL_INVOCATION_INTERVAL_ALPHA) * (now - timestamp_last_control_invocation);
  timestamp_last_control_invocation = now;
  bool set_motors = (now - timestamp_last_control_packet_received < CONTROL_PACKET_TIMEOUT_USEC)  || (set_motors_overwrite == 1 && motor_cmd_divider >= 3);
  log_set_motors = set_motors ? 1 : 0;
  set_backprop_tools_overwrite_stabilizer(set_motors);
  if(!prev_set_motors && set_motors){
    timestamp_last_waypoint = now;
    timestamp_controller_activation = now;
    waypoint_navigation_dynamic_current_waypoint = 0;
    origin[0] = state->position.x;
    origin[1] = state->position.y;
    origin[2] = state->position.z + target_height;
    figure_eight_last_invocation = now;
    figure_eight_progress = 0;
    controllerMellingerFirmwareInit();
    controllerINDIInit();
    controllerMellingerFirmwareEnableIntegrators(MELLINGER_ENABLE_INTEGRATORS == 1);
    DEBUG_PRINT("Controller activated\n");
    switch(mode){
      case POSITION:
        DEBUG_PRINT("POSITION mode\n");
        break;
      case WAYPOINT_NAVIGATION:
        DEBUG_PRINT("WAYPOINT_NAVIGATION mode\n");
        break;
      case WAYPOINT_NAVIGATION_DYNAMIC:
        DEBUG_PRINT("WAYPOINT_NAVIGATION_DYNAMIC mode\n");
        break;
      case FIGURE_EIGHT:
        DEBUG_PRINT("FIGURE_EIGHT mode\n");
        break;
    }
  }
  if(prev_set_motors && !set_motors){
    DEBUG_PRINT("Controller deactivated\n");
    for(uint8_t i=0; i<4; i++){
      motorsSetRatio(motors[i], 0);
    }
  }
  relative_pos[0] = state->position.x - origin[0];
  relative_pos[1] = state->position.y - origin[1];
  relative_pos[2] = state->position.z - origin[2];
  target_vel[0] = 0;
  target_vel[1] = 0;
  target_vel[2] = 0;
  switch(mode){
    case POSITION:
      target_pos[0] = origin[0];
      target_pos[1] = origin[1];
      target_pos[2] = origin[2];
      break;
    case WAYPOINT_NAVIGATION:
    {
      uint64_t elapsed_since_start = (now-timestamp_last_waypoint);
      int current_point = (elapsed_since_start / WAYPOINT_NAVIGATION_POINT_DURATION) % WAYPOINT_NAVIGATION_POINTS;
      target_pos[0] = trajectory[current_point][0] * trajectory_scale + origin[0];
      target_pos[1] = trajectory[current_point][1] * trajectory_scale + origin[1];
      target_pos[2] = trajectory[current_point][2] * trajectory_scale + origin[2];
    }
      break;
    case WAYPOINT_NAVIGATION_DYNAMIC:
      {
        float x = relative_pos[0] - trajectory[waypoint_navigation_dynamic_current_waypoint][0];
        float y = relative_pos[1] - trajectory[waypoint_navigation_dynamic_current_waypoint][1];
        float z = relative_pos[2] - trajectory[waypoint_navigation_dynamic_current_waypoint][2];

        float current_dist = sqrtf(x*x + y*y + z*z);
        if(current_dist < waypoint_navigation_dynamic_threshold){
          waypoint_navigation_dynamic_current_waypoint = (waypoint_navigation_dynamic_current_waypoint + 1) % WAYPOINT_NAVIGATION_POINTS;
          DEBUG_PRINT("Next waypoint %d, [%f, %f, %f]\n", waypoint_navigation_dynamic_current_waypoint, trajectory[waypoint_navigation_dynamic_current_waypoint][0], trajectory[waypoint_navigation_dynamic_current_waypoint][1], trajectory[waypoint_navigation_dynamic_current_waypoint][2]);
        }
        target_pos[0] = origin[0] + trajectory[waypoint_navigation_dynamic_current_waypoint][0];
        target_pos[1] = origin[1] + trajectory[waypoint_navigation_dynamic_current_waypoint][1];
        target_pos[2] = origin[2] + trajectory[waypoint_navigation_dynamic_current_waypoint][2];
      }
      break;
    case FIGURE_EIGHT:
      {
        float t = (now - timestamp_controller_activation) / 1000000.0f;
        float dt = (now - figure_eight_last_invocation) / 1000000.0f;
        float target_speed = 1/figure_eight_interval;
        float speed = target_speed;
        if(t < figure_eight_warmup_time){
          speed = target_speed * t/figure_eight_warmup_time;
        }
        figure_eight_progress += dt * speed;
        float progress = figure_eight_progress;
        target_pos[0] = origin[0] + cosf(progress*2*M_PI + M_PI / 2) * figure_eight_scale;
        target_vel[0] = -sinf(progress*2*M_PI + M_PI / 2) * figure_eight_scale * 2 * M_PI * speed;
        target_pos[1] = origin[1] + sinf(2*(progress*2*M_PI + M_PI / 2)) / 2.0f * figure_eight_scale;
        target_vel[1] = cosf(2*(progress*2*M_PI + M_PI / 2)) / 2.0f * figure_eight_scale * 4 * M_PI * speed;
        target_pos[2] = origin[2];
        figure_eight_last_invocation = now;
      }
      break;
  }
  pos_error[0] = target_pos[0] - state->position.x;
  pos_error[1] = target_pos[1] - state->position.y;
  pos_error[2] = target_pos[2] - state->position.z;

  trigger_every(controller_tick);
  prev_set_motors = set_motors;

  if (tick % CONTROL_INTERVAL_MS == 0){
    update_state(sensors, state);
    {
      int64_t before = usecTimestamp();
      if(use_orig_controller == 0){
        backprop_tools_control(state_input, action_output);
      }
      else{
        action_output[0] = -0.8;
        action_output[1] = -0.8;
        action_output[2] = -0.8;
        action_output[3] = -0.8;
      }
      int64_t after = usecTimestamp();
      if (tick % (CONTROL_INTERVAL_MS * 10000) == 0){
        DEBUG_PRINT("backprop_tools_control took %lldus\n", after - before);
      }
    }
    for(uint8_t i=0; i<4; i++){
      if (tick % (CONTROL_INTERVAL_MS * 10000) == 0){
        DEBUG_PRINT("action_output[%d]: %f\n", i, action_output[i]);
      }
      float a_pp = (action_output[i] + 1)/2;
      float des_rpm = (MAX_RPM - MIN_RPM) * a_pp + MIN_RPM;
      float des_percentage = des_rpm / MAX_RPM;
      motor_cmd[i] = des_percentage * UINT16_MAX;
      if(set_motors && use_orig_controller == 0){
        motorsSetRatio(motors[i], clip((float)motor_cmd[i] / motor_cmd_divider, 0, UINT16_MAX));
      }
    }
    int64_t spare_time = CONTROL_INTERVAL_US - (now - timestamp_last_reset) ;
    if(spare_time < 0 && (now - timestamp_last_behind_schedule_message > BEHIND_SCHEDULE_MESSAGE_MIN_INTERVAL)){
      DEBUG_PRINT("Learned Controller is behind schedule: %lldus/%dus\n", (int64_t)(now-timestamp_last_reset), CONTROL_INTERVAL_US);
      timestamp_last_behind_schedule_message = now;
    }
    timestamp_last_reset = usecTimestamp();
  }
  if(!set_motors){
    controllerPid(control, setpoint, sensors, state, tick);
    powerDistribution(control, &motorThrustUncapped);
    batteryCompensation(&motorThrustUncapped, &motorThrustBatCompUncapped);
    powerDistributionCap(&motorThrustBatCompUncapped, &motorPwm);
    setMotorRatios(&motorPwm);
  }
  else{
    if(use_orig_controller >= 1){
      setpoint->mode.x = modeAbs;
      setpoint->mode.y = modeAbs;
      setpoint->mode.z = modeAbs;
      setpoint->mode.yaw = modeAbs;
      setpoint->mode.pitch = modeDisable;
      setpoint->mode.roll = modeDisable;
      setpoint->mode.quat = modeDisable;
      setpoint->position.x = target_pos[0];
      setpoint->position.y = target_pos[1];
      setpoint->position.z = target_pos[2];
      setpoint->velocity.x = target_vel[0];
      setpoint->velocity.y = target_vel[1];
      setpoint->velocity.z = target_vel[2];
      setpoint->acceleration.x = 0;
      setpoint->acceleration.y = 0;
      setpoint->acceleration.z = 0;

      setpoint->attitude.yaw = 0;
      setpoint->attitude.pitch = 0;
      setpoint->attitude.roll = 0;
      setpoint->attitudeQuaternion.w = 1;
      setpoint->attitudeQuaternion.x = 0;
      setpoint->attitudeQuaternion.y = 0;
      setpoint->attitudeQuaternion.z = 0;
      setpoint->attitudeRate.yaw = 0;
      setpoint->attitudeRate.pitch = 0;
      setpoint->attitudeRate.roll = 0;

      setpoint->timestamp = xTaskGetTickCount();
      if(use_orig_controller == 1){
        controllerPid(control, setpoint, sensors, state, tick);
      }
      else{
        if(use_orig_controller == 2){
          setpoint->position.x = state->position.x + clip(target_pos[0] - state->position.x, -POS_DISTANCE_LIMIT_MELLINGER, POS_DISTANCE_LIMIT_MELLINGER);
          setpoint->position.y = state->position.y + clip(target_pos[1] - state->position.y, -POS_DISTANCE_LIMIT_MELLINGER, POS_DISTANCE_LIMIT_MELLINGER);
          setpoint->position.z = state->position.z + clip(target_pos[2] - state->position.z, -POS_DISTANCE_LIMIT_MELLINGER, POS_DISTANCE_LIMIT_MELLINGER);
          setpoint->velocity.x = state->velocity.x + clip(target_vel[0] - state->velocity.x, -VEL_DISTANCE_LIMIT_MELLINGER, VEL_DISTANCE_LIMIT_MELLINGER);
          setpoint->velocity.y = state->velocity.y + clip(target_vel[1] - state->velocity.y, -VEL_DISTANCE_LIMIT_MELLINGER, VEL_DISTANCE_LIMIT_MELLINGER);
          setpoint->velocity.z = state->velocity.z + clip(target_vel[2] - state->velocity.z, -VEL_DISTANCE_LIMIT_MELLINGER, VEL_DISTANCE_LIMIT_MELLINGER);
          controllerMellingerFirmware(control, setpoint, sensors, state, tick);
        }
        else{
          if(use_orig_controller == 3){
            controllerINDI(control, setpoint, sensors, state, tick);
          }
          else{
            setpoint->position.x = state->position.x + clip(target_pos[0] - state->position.x, -POS_DISTANCE_LIMIT_BRESCIANI, POS_DISTANCE_LIMIT_BRESCIANI);
            setpoint->position.y = state->position.y + clip(target_pos[1] - state->position.y, -POS_DISTANCE_LIMIT_BRESCIANI, POS_DISTANCE_LIMIT_BRESCIANI);
            setpoint->position.z = state->position.z + clip(target_pos[2] - state->position.z, -POS_DISTANCE_LIMIT_BRESCIANI, POS_DISTANCE_LIMIT_BRESCIANI);
            setpoint->velocity.x = state->velocity.x + clip(target_vel[0] - state->velocity.x, -VEL_DISTANCE_LIMIT_BRESCIANI, VEL_DISTANCE_LIMIT_BRESCIANI);
            setpoint->velocity.y = state->velocity.y + clip(target_vel[1] - state->velocity.y, -VEL_DISTANCE_LIMIT_BRESCIANI, VEL_DISTANCE_LIMIT_BRESCIANI);
            setpoint->velocity.z = state->velocity.z + clip(target_vel[2] - state->velocity.z, -VEL_DISTANCE_LIMIT_BRESCIANI, VEL_DISTANCE_LIMIT_BRESCIANI);
            controllerBrescianini(control, setpoint, sensors, state, tick);
          }
        }
      }
      powerDistribution(control, &motorThrustUncapped);
      batteryCompensation(&motorThrustUncapped, &motorThrustBatCompUncapped);
      powerDistributionCap(&motorThrustBatCompUncapped, &motorPwm);
      setMotorRatios(&motorPwm);
    }
  }
  controller_tick++;
}



PARAM_GROUP_START(bpt)
PARAM_ADD(PARAM_FLOAT, motor_div, &motor_cmd_divider)
PARAM_ADD(PARAM_FLOAT, target_z, &target_height)
PARAM_ADD(PARAM_UINT8, smo, &set_motors_overwrite)
PARAM_ADD(PARAM_UINT8, ht, &hand_test)
PARAM_ADD(PARAM_UINT8, wn, &mode)
PARAM_ADD(PARAM_FLOAT, ts, &trajectory_scale)
PARAM_ADD(PARAM_FLOAT, wpt, &waypoint_navigation_dynamic_threshold)
PARAM_ADD(PARAM_FLOAT, fewt, &figure_eight_warmup_time)
PARAM_ADD(PARAM_FLOAT, fei, &figure_eight_interval)
PARAM_ADD(PARAM_FLOAT, fes, &figure_eight_scale)
PARAM_ADD(PARAM_FLOAT, pdl, &POS_DISTANCE_LIMIT)
PARAM_ADD(PARAM_FLOAT, vdl, &VEL_DISTANCE_LIMIT)
PARAM_ADD(PARAM_FLOAT, pdlm, &POS_DISTANCE_LIMIT_MELLINGER)
PARAM_ADD(PARAM_FLOAT, vdlm, &VEL_DISTANCE_LIMIT_MELLINGER)
PARAM_ADD(PARAM_UINT8, orig, &use_orig_controller)
PARAM_ADD(PARAM_UINT8, mei, &MELLINGER_ENABLE_INTEGRATORS)
PARAM_GROUP_STOP(bpt)


// LOG_GROUP_START(bptp)
// LOG_ADD(LOG_FLOAT, x, &state_input[0])
// LOG_ADD(LOG_FLOAT, y, &state_input[1])
// LOG_ADD(LOG_FLOAT, z, &state_input[2])
// LOG_GROUP_STOP(bptp)

// LOG_GROUP_START(bptq)
// LOG_ADD(LOG_FLOAT, w, &state_input[3])
// LOG_ADD(LOG_FLOAT, x, &state_input[4])
// LOG_ADD(LOG_FLOAT, y, &state_input[5])
// LOG_ADD(LOG_FLOAT, z, &state_input[6])
// LOG_GROUP_STOP(bptq)

// LOG_GROUP_START(bpttwl)
// LOG_ADD(LOG_FLOAT, x, &state_input[7])
// LOG_ADD(LOG_FLOAT, y, &state_input[8])
// LOG_ADD(LOG_FLOAT, z, &state_input[9])
// LOG_GROUP_STOP(bpttwl)

// LOG_GROUP_START(bpttwa)
// LOG_ADD(LOG_FLOAT, x, &state_input[10])
// LOG_ADD(LOG_FLOAT, y, &state_input[11])
// LOG_ADD(LOG_FLOAT, z, &state_input[12])
// LOG_GROUP_STOP(bpttwa)

// LOG_GROUP_START(bptt)
// LOG_ADD(LOG_FLOAT, x, &target_pos[0])
// LOG_ADD(LOG_FLOAT, y, &target_pos[1])
// LOG_ADD(LOG_FLOAT, z, &target_pos[2])
// LOG_GROUP_STOP(bptt)


LOG_GROUP_START(bptm)
LOG_ADD(LOG_UINT16, m1, &motor_cmd[0])
LOG_ADD(LOG_UINT16, m2, &motor_cmd[1])
LOG_ADD(LOG_UINT16, m3, &motor_cmd[2])
LOG_ADD(LOG_UINT16, m4, &motor_cmd[3])
LOG_GROUP_STOP(bptm)

LOG_GROUP_START(bptrp)
LOG_ADD(LOG_FLOAT, x, &relative_pos[0])
LOG_ADD(LOG_FLOAT, y, &relative_pos[1])
LOG_ADD(LOG_FLOAT, z, &relative_pos[2])
LOG_ADD(LOG_UINT8, sm, &log_set_motors)
LOG_GROUP_STOP(bptrp)

LOG_GROUP_START(bpttp)
LOG_ADD(LOG_FLOAT, x, &target_pos[0])
LOG_ADD(LOG_FLOAT, y, &target_pos[1])
LOG_ADD(LOG_FLOAT, z, &target_pos[2])
LOG_GROUP_STOP(bptrp)

LOG_GROUP_START(bptte)
LOG_ADD(LOG_FLOAT, x, &pos_error[0])
LOG_ADD(LOG_FLOAT, y, &pos_error[1])
LOG_ADD(LOG_FLOAT, z, &pos_error[2])
LOG_GROUP_STOP(bptre)

