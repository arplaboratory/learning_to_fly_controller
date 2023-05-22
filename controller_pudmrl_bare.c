
#include "controller_pudmrl.h"
#include "debug.h"
#include "usec_time.h"
#include "policy_generated.h"
#include <math.h>
#include "math3d.h"
#include "log.h"
#include "param.h"
#include "motors.h"

// #define DEBUG_OUTPUT_INTERVAL 100
#define CONTROL_INTERVAL_MS 20
#define FORWARD_STEPS_PER_ITERATION 10000

void controllerPudmrlInit(void)
{
  set_forward_state(0);
  set_input();
  // DEBUG_PRINT("PUDM-RL: Init\n");
}

bool controllerPudmrlTest(void)
{
  return true;
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
  v[0] = atan2f( m[3*1 + 0],m[3*0 + 0]);
  v[1] = atan2f(-m[3*2 + 0],sqrtf(m[3*2 + 1]*m[3*2 + 1]+m[3*2 + 2]*m[3*2 + 2]));
  v[2] = atan2f( m[3*2 + 1],m[3*2 + 2]);
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


void controllerPudmrl(control_t *control, setpoint_t *setpoint,
                                         const sensorData_t *sensors,
                                         const state_t *state,
                                         const uint32_t tick)
{
  if(tick > 5000 && tick % 1000 == 55){

    bool finished = true;
    uint64_t before = usecTimestamp();
    set_input();
    forward();
    uint64_t after = usecTimestamp();
    DEBUG_PRINT("forward took %lld\n", after-before);
    if (finished){
      controllerPudmrlInit();
    }
  }
}