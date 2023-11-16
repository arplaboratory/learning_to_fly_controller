#include "debug.h"
#include "usec_time.h"
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
#include "rl_tools_adapter.h"
#include "stabilizer_types.h"
#include "pm.h"
#include "task.h"

#define CONTROL_INTERVAL_MS 2
#define CONTROL_INTERVAL_US (CONTROL_INTERVAL_MS * 1000)
#define CONTROL_PACKET_TIMEOUT_USEC (1000*400)
#define BEHIND_SCHEDULE_MESSAGE_MIN_INTERVAL (1000000)
#define CONTROL_INVOCATION_INTERVAL_ALPHA 0.95f
#define DEBUG_MEASURE_FORWARD_TIME
#define MIN_RPM 0
#define MAX_RPM 21702.1
#define WAYPOINT_NAVIGATION_NUMBER_OF_POINTS (5)
#define WARMUP_TIME (1000 * 500)
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
static uint64_t timestamp_last_control_packet_received_hover = 0;
static uint64_t timestamp_controller_activation;
static uint64_t timestamp_pre_set_motors;

// Logging variables
static float control_invocation_interval = 0;

// Control variables: input
static float target_pos[3] = {0, 0, 0};
static float target_vel[3] = {0, 0, 0};
static float pos_error[3] = {0, 0, 0};
static float relative_pos[3] = {0, 0, 0};
static float origin[3] = {0, 0, 0};

static float pos_distance_limit_position;
static float vel_distance_limit_position;
static float pos_distance_limit_figure_eight;
static float vel_distance_limit_figure_eight;
static float pos_distance_limit_mellinger;
static float vel_distance_limit_mellinger;
static float pos_distance_limit_bresciani;
static float vel_distance_limit_bresciani;
static uint8_t mellinger_enable_integrators;
static uint8_t log_set_motors = 0;
static float velocity_cmd_multiplier, velocity_cmd_p_term;

enum Mode{
  NORMAL = 0,
  POSITION = 1,
  WAYPOINT_NAVIGATION = 2,
  WAYPOINT_NAVIGATION_DYNAMIC = 3,
  FIGURE_EIGHT = 4
};
enum TriggerMode{
  RL_TOOLS_PACKET = 0,
  HOVER_PACKET = 1,
};
static uint8_t mode;
static uint8_t trigger_mode;
static float trajectory[WAYPOINT_NAVIGATION_NUMBER_OF_POINTS][3] = {
  {0.0, 0.0, 0.0},
  {1.0, 0.0, 0.0},
  {1.0, 1.0, 0.0},
  {0.0, 1.0, 0.0},
  {0.0, 0.0, 0.0},
};
static float target_height;

static uint64_t waypoint_navigation_timestamp_start;
static uint8_t  waypoint_navigation_dynamic_current_waypoint = 0;
static float    waypoint_navigation_dynamic_threshold = 0.1;
static float    waypoint_navigation_point_duration = 4;
static float    waypoint_navigation_trajectory_scale = 0.5;

static float    figure_eight_interval = 5.5;
static float    figure_eight_warmup_time; 
static float    figure_eight_scale = 1.0;
static float    figure_eight_progress = 0;
static uint64_t figure_eight_last_invocation;
static float    target_height_figure_eight;

static float state_input[13];
static float action_output[4];

const uint8_t motors[4] = {MOTOR_M1, MOTOR_M2, MOTOR_M3, MOTOR_M4};
static uint8_t set_motors_overwrite = 0;
static uint16_t motor_cmd[4];
static float motor_cmd_divider, motor_cmd_divider_warmup;
static bool prev_set_motors, prev_pre_set_motors;

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
    float POS_DISTANCE_LIMIT = mode == FIGURE_EIGHT ? pos_distance_limit_figure_eight : pos_distance_limit_position;
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
    float VEL_DISTANCE_LIMIT = mode == FIGURE_EIGHT ? vel_distance_limit_figure_eight : vel_distance_limit_position;
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

void rl_tools_controller_packet_received(){
  uint64_t now = usecTimestamp();
  timestamp_last_control_packet_received = now;
}
// void rl_tools_controller_hover_packet_received(){
//   uint64_t now = usecTimestamp();
//   timestamp_last_control_packet_received_hover = now;
//   DEBUG_PRINT("Hover packet received\n");
// }


void controllerOutOfTreeInit(void){
  controller_state = STATE_RESET;
  controller_tick = 0;
  motor_cmd_divider = 1.0;
  motor_cmd_divider_warmup = 7.0;
  motor_cmd[0] = 0;
  motor_cmd[1] = 0;
  motor_cmd[2] = 0;
  motor_cmd[3] = 0;
  timestamp_last_reset = usecTimestamp();
  prev_set_motors = false;
  prev_pre_set_motors = false;
  timestamp_last_control_packet_received = 0;
  timestamp_last_control_packet_received_hover = 0;
  timestamp_last_behind_schedule_message = 0;
  control_invocation_interval = 0;
  forward_tick = 0;
  hand_test = 0;
  waypoint_navigation_timestamp_start = 0;
  waypoint_navigation_trajectory_scale = 0.5;
  relative_pos[0] = 0;
  relative_pos[1] = 0;
  relative_pos[2] = 0;
  log_set_motors = 0;

  pos_distance_limit_position = 0.5f;
  vel_distance_limit_position = 2.0f;
  pos_distance_limit_figure_eight = 0.6f;
  vel_distance_limit_figure_eight = 2.0f;
  pos_distance_limit_mellinger = 0.2f;
  vel_distance_limit_mellinger = 1.0f;
  pos_distance_limit_bresciani = 0.2f;
  vel_distance_limit_bresciani = 1.0f;
  mellinger_enable_integrators = 1;
  velocity_cmd_multiplier = 1;
  velocity_cmd_p_term = 0.0;

  target_height = 0.3;
  target_height_figure_eight = 0.0;

  // mode = NORMAL;
  mode = POSITION;
  // mode = FIGURE_EIGHT;
  // trigger_mode = RL_TOOLS_PACKET;
  trigger_mode = HOVER_PACKET;
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
  rl_tools_init();

  DEBUG_PRINT("BackpropTools controller init! Checkpoint: %s\n", rl_tools_get_checkpoint_name());
}

bool controllerOutOfTreeTest(void)
{
  float output[4];
  float absdiff = rl_tools_test(output);
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

  DEBUG_PRINT("Last setpoint: x disposition/mode %f/%f/%d\n", last_setpoint.position.x, last_setpoint.velocity.x, last_setpoint.mode.x);
  DEBUG_PRINT("Last setpoint: y disposition/mode %f/%f/%d\n", last_setpoint.position.y, last_setpoint.velocity.y, last_setpoint.mode.y);
  DEBUG_PRINT("Last setpoint: z disposition/mode %f/%f/%d\n", last_setpoint.position.z, last_setpoint.velocity.z, last_setpoint.mode.z);
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

static void print_mode(stab_mode_t mode){
  switch(mode){
    case modeDisable:
      DEBUG_PRINT("modeDisable\n");
      break;
    case modeAbs:
      DEBUG_PRINT("modeAbs\n");
      break;
    case modeVelocity:
      DEBUG_PRINT("modeVelocity\n");
      break;
  }
}

void controllerOutOfTree(control_t *control, setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  uint64_t now = usecTimestamp();
  if(setpoint->mode.x == modeVelocity && setpoint->mode.y == modeVelocity){
    timestamp_last_control_packet_received_hover = now;
  }

  last_setpoint = *setpoint;
  watchdogReset();
  control_invocation_interval *= CONTROL_INVOCATION_INTERVAL_ALPHA;
  control_invocation_interval += (1-CONTROL_INVOCATION_INTERVAL_ALPHA) * (now - timestamp_last_control_invocation);
  timestamp_last_control_invocation = now;
  uint64_t relevant_timestamp_last_control_packet_received = trigger_mode == RL_TOOLS_PACKET ? timestamp_last_control_packet_received : timestamp_last_control_packet_received_hover;
  bool pre_set_motors = (now - relevant_timestamp_last_control_packet_received < CONTROL_PACKET_TIMEOUT_USEC)  || (set_motors_overwrite == 1 && motor_cmd_divider >= 3);
  bool set_motors = false;

  if(!prev_pre_set_motors && pre_set_motors){
    timestamp_pre_set_motors = now;
  }
  set_motors = pre_set_motors && ((now - timestamp_pre_set_motors) > WARMUP_TIME);

  log_set_motors = set_motors ? 1 : 0;
  // set_rl_tools_overwrite_stabilizer(set_motors);
  if(!prev_set_motors && set_motors){
    waypoint_navigation_timestamp_start = now;
    timestamp_controller_activation = now;
    waypoint_navigation_dynamic_current_waypoint = 0;
    origin[0] = state->position.x;
    origin[1] = state->position.y;
    origin[2] = state->position.z + (mode == FIGURE_EIGHT ? target_height_figure_eight : target_height);
    figure_eight_last_invocation = now;
    figure_eight_progress = 0;
    controllerMellingerFirmwareInit();
    controllerINDIInit();
    // controllerMellingerFirmwareEnableIntegrators(MELLINGER_ENABLE_INTEGRATORS == 1);
    DEBUG_PRINT("Controller activated\n");
    switch(mode){
      case NORMAL:
        DEBUG_PRINT("NORMAL mode \n");
        DEBUG_PRINT("\t x mode: "); print_mode(setpoint->mode.x);
        DEBUG_PRINT("\t y mode: "); print_mode(setpoint->mode.y);
        DEBUG_PRINT("\t z mode: "); print_mode(setpoint->mode.z);
        break;
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
    case NORMAL:
      switch(setpoint->mode.x){
        case modeAbs:
        target_pos[0] = setpoint->position.x;
        target_vel[0] = 0;
        break;
        case modeVelocity:
        target_pos[0] = state->position.x - setpoint->velocity.x * velocity_cmd_p_term;
        target_vel[0] = setpoint->velocity.x * velocity_cmd_multiplier;
        break;
        case modeDisable:
        target_pos[0] = origin[0];
        target_vel[0] = 0;
        break;
      }
      switch(setpoint->mode.y){
        case modeAbs:
        target_pos[1] = setpoint->position.y;
        target_vel[1] = 0;
        break;
        case modeVelocity:
        target_pos[1] = state->position.y - setpoint->velocity.y * velocity_cmd_p_term;
        target_vel[1] = setpoint->velocity.y * velocity_cmd_multiplier;
        break;
        case modeDisable:
        target_pos[1] = origin[1];
        target_vel[1] = 0;
        break;
      }
      switch(setpoint->mode.z){
        case modeAbs:
        target_pos[2] = setpoint->position.z;
        target_vel[2] = 0;
        break;
        case modeVelocity:
        target_pos[2] = state->position.z - setpoint->velocity.z * velocity_cmd_p_term;
        target_vel[2] = setpoint->velocity.z * velocity_cmd_multiplier;
        break;
        case modeDisable:
        target_pos[2] = origin[2];
        target_vel[2] = 0;
        break;
      }
    break;
    case POSITION:
      target_pos[0] = origin[0];
      target_pos[1] = origin[1];
      target_pos[2] = origin[2];
      break;
    case WAYPOINT_NAVIGATION:
    {
      uint64_t elapsed_since_start = (now-waypoint_navigation_timestamp_start);
      int current_point = (elapsed_since_start / ((int)(waypoint_navigation_point_duration * 1000 * 1000))) % WAYPOINT_NAVIGATION_NUMBER_OF_POINTS;
      target_pos[0] = trajectory[current_point][0] * waypoint_navigation_trajectory_scale + origin[0];
      target_pos[1] = trajectory[current_point][1] * waypoint_navigation_trajectory_scale + origin[1];
      target_pos[2] = trajectory[current_point][2] * waypoint_navigation_trajectory_scale + origin[2];
    }
      break;
    case WAYPOINT_NAVIGATION_DYNAMIC:
      {
        float x = relative_pos[0] - trajectory[waypoint_navigation_dynamic_current_waypoint][0];
        float y = relative_pos[1] - trajectory[waypoint_navigation_dynamic_current_waypoint][1];
        float z = relative_pos[2] - trajectory[waypoint_navigation_dynamic_current_waypoint][2];

        float current_dist = sqrtf(x*x + y*y + z*z);
        if(current_dist < waypoint_navigation_dynamic_threshold){
          waypoint_navigation_dynamic_current_waypoint = (waypoint_navigation_dynamic_current_waypoint + 1) % WAYPOINT_NAVIGATION_NUMBER_OF_POINTS;
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
  prev_pre_set_motors = pre_set_motors;

  if(tick % CONTROL_INTERVAL_MS == 0){
    update_state(sensors, state);
    {
      int64_t before = usecTimestamp();
      if(use_orig_controller == 0){
        rl_tools_control(state_input, action_output);
      }
      else{
        action_output[0] = -0.8;
        action_output[1] = -0.8;
        action_output[2] = -0.8;
        action_output[3] = -0.8;
      }
      int64_t after = usecTimestamp();
      if (tick % (CONTROL_INTERVAL_MS * 10000) == 0){
        DEBUG_PRINT("rl_tools_control took %lldus\n", after - before);
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
    if(pre_set_motors){
      for(uint8_t i=0; i<4; i++){
        motorsSetRatio(motors[i], UINT16_MAX / motor_cmd_divider_warmup);
      }
    }
    else{
      controllerPid(control, setpoint, sensors, state, tick);
      powerDistribution(control, &motorThrustUncapped);
      batteryCompensation(&motorThrustUncapped, &motorThrustBatCompUncapped);
      powerDistributionCap(&motorThrustBatCompUncapped, &motorPwm);
      setMotorRatios(&motorPwm);
    }
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
          setpoint->position.x = state->position.x + clip(target_pos[0] - state->position.x, -pos_distance_limit_mellinger, pos_distance_limit_mellinger);
          setpoint->position.y = state->position.y + clip(target_pos[1] - state->position.y, -pos_distance_limit_mellinger, pos_distance_limit_mellinger);
          setpoint->position.z = state->position.z + clip(target_pos[2] - state->position.z, -pos_distance_limit_mellinger, pos_distance_limit_mellinger);
          setpoint->velocity.x = state->velocity.x + clip(target_vel[0] - state->velocity.x, -vel_distance_limit_mellinger, vel_distance_limit_mellinger);
          setpoint->velocity.y = state->velocity.y + clip(target_vel[1] - state->velocity.y, -vel_distance_limit_mellinger, vel_distance_limit_mellinger);
          setpoint->velocity.z = state->velocity.z + clip(target_vel[2] - state->velocity.z, -vel_distance_limit_mellinger, vel_distance_limit_mellinger);
          controllerMellingerFirmware(control, setpoint, sensors, state, tick);
        }
        else{
          if(use_orig_controller == 3){
            controllerINDI(control, setpoint, sensors, state, tick);
          }
          else{
            setpoint->position.x = state->position.x + clip(target_pos[0] - state->position.x, -pos_distance_limit_bresciani, pos_distance_limit_bresciani);
            setpoint->position.y = state->position.y + clip(target_pos[1] - state->position.y, -pos_distance_limit_bresciani, pos_distance_limit_bresciani);
            setpoint->position.z = state->position.z + clip(target_pos[2] - state->position.z, -pos_distance_limit_bresciani, pos_distance_limit_bresciani);
            setpoint->velocity.x = state->velocity.x + clip(target_vel[0] - state->velocity.x, -vel_distance_limit_bresciani, vel_distance_limit_bresciani);
            setpoint->velocity.y = state->velocity.y + clip(target_vel[1] - state->velocity.y, -vel_distance_limit_bresciani, vel_distance_limit_bresciani);
            setpoint->velocity.z = state->velocity.z + clip(target_vel[2] - state->velocity.z, -vel_distance_limit_bresciani, vel_distance_limit_bresciani);
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



PARAM_GROUP_START(rlt)
PARAM_ADD(PARAM_UINT8, trigger, &trigger_mode)
PARAM_ADD(PARAM_FLOAT, motor_div, &motor_cmd_divider)
PARAM_ADD(PARAM_FLOAT, motor_div_wu, &motor_cmd_divider_warmup)
PARAM_ADD(PARAM_FLOAT, target_z, &target_height)
PARAM_ADD(PARAM_FLOAT, target_z_fe, &target_height_figure_eight)
PARAM_ADD(PARAM_UINT8, smo, &set_motors_overwrite)
PARAM_ADD(PARAM_UINT8, ht, &hand_test)
PARAM_ADD(PARAM_UINT8, wn, &mode)
PARAM_ADD(PARAM_FLOAT, ts, &waypoint_navigation_trajectory_scale)
PARAM_ADD(PARAM_FLOAT, wpt, &waypoint_navigation_dynamic_threshold)
PARAM_ADD(PARAM_FLOAT, wni, &waypoint_navigation_point_duration)
PARAM_ADD(PARAM_FLOAT, fewt, &figure_eight_warmup_time)
PARAM_ADD(PARAM_FLOAT, fei,  &figure_eight_interval)
PARAM_ADD(PARAM_FLOAT, fes,  &figure_eight_scale)
PARAM_ADD(PARAM_FLOAT, pdlp,  &pos_distance_limit_position)
PARAM_ADD(PARAM_FLOAT, pdlfe, &pos_distance_limit_figure_eight)
PARAM_ADD(PARAM_FLOAT, vdlp,  &vel_distance_limit_position)
PARAM_ADD(PARAM_FLOAT, vdlfe, &vel_distance_limit_figure_eight)
PARAM_ADD(PARAM_FLOAT, pdlm,  &pos_distance_limit_mellinger)
PARAM_ADD(PARAM_FLOAT, vdlm,  &vel_distance_limit_mellinger)
PARAM_ADD(PARAM_UINT8, orig, &use_orig_controller)
PARAM_ADD(PARAM_UINT8, mei, &mellinger_enable_integrators)
PARAM_ADD(PARAM_FLOAT, vcmdm, &velocity_cmd_multiplier)
PARAM_ADD(PARAM_FLOAT, vcmdp, &velocity_cmd_p_term)
PARAM_GROUP_STOP(rlt)


LOG_GROUP_START(rltm)
LOG_ADD(LOG_UINT16, m1, &motor_cmd[0])
LOG_ADD(LOG_UINT16, m2, &motor_cmd[1])
LOG_ADD(LOG_UINT16, m3, &motor_cmd[2])
LOG_ADD(LOG_UINT16, m4, &motor_cmd[3])
LOG_GROUP_STOP(rltm)

LOG_GROUP_START(rltrp)
LOG_ADD(LOG_FLOAT, x, &relative_pos[0])
LOG_ADD(LOG_FLOAT, y, &relative_pos[1])
LOG_ADD(LOG_FLOAT, z, &relative_pos[2])
LOG_ADD(LOG_UINT8, sm, &log_set_motors)
LOG_GROUP_STOP(rltrp)

LOG_GROUP_START(rlttp)
LOG_ADD(LOG_FLOAT, x, &target_pos[0])
LOG_ADD(LOG_FLOAT, y, &target_pos[1])
LOG_ADD(LOG_FLOAT, z, &target_pos[2])
LOG_GROUP_STOP(rltrp)

LOG_GROUP_START(rltte)
LOG_ADD(LOG_FLOAT, x, &pos_error[0])
LOG_ADD(LOG_FLOAT, y, &pos_error[1])
LOG_ADD(LOG_FLOAT, z, &pos_error[2])
LOG_GROUP_STOP(rltre)

