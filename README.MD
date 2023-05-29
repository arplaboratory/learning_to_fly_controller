The CrazyFlie firmware is expected in `../crazyflie-firmware`

Set the default controller type to be OOT (alternatively set the `stabilizer.controller` parameter to `5`):
```diff
-#define DEFAULT_CONTROLLER ControllerTypePID
+#define DEFAULT_CONTROLLER ControllerTypeOot
```

In `../crazyflie-firmware` run
```
make cf2_defconfig
```
first, then
```
make
```
in this folder.


The main `crazyflie-firmware` has been changed to be in `DEBUG` mode and the `stabilizer.c` motor commands have been disabled


Flashing:
```
CLOAD_CMDS="-w radio://0/80/2M" make cload
```


Vicon
```
Run the vicon client:
```
rosrun mocap_vicon vicon.launch
```
Run the ros to websocket bridge
```
roslaunch rosbridge_server rosbridge_websocket.launch
```
Run the modified `crazyflie-clients-python` (this is forwarding the position from vicon to the crazyflie)
Modify the topic in the main.py to reflect the vicon object that corresponds to the crazyflie.
```
cfclient
```
Set the `stabilizer.estimator` parameter to EKF so that the external position reference is incorporated into the estimate (this is set by default in the modified fw now)


Test the angular rate response using the ht (hand test) parameter in the firmware (check the hand_test var in the controller code to see the options)