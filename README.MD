The CrazyFlie firmware is expected in `../crazyflie-firmware`

Set the default controller type to be OOT (alternatively set the `stabilizer.controller` parameter to `5`):
```diff
-#define DEFAULT_CONTROLLER ControllerTypePID
+#define DEFAULT_CONTROLLER ControllerTypeOot
```


# Building & Flashing
## Build using the bitcraze/builder Docker image
Pull submodules of the firmward recursively
```
git submodule update --init --recursive -- external/crazyflie_firmware  
```

Configure the firmware
```
docker run --rm -v $(cd external/crazyflie_firmware && pwd):/module bitcraze/builder make cf2_defconfig
```

Build OOT controller and firmware
```
docker run --rm -it -v $(pwd):/module bitcraze/builder bash -c "make -j8"
```

Flash natively
```
CLOAD_CMDS="-w radio://0/80/2M" make cload
```

Note the cload script is needed on Apple Silicon Macs because the x86_64 version of Python and the cfclient library are required for flashing using the CrazyRadio

```
CLOAD_CMDS="-w radio://0/80/2M" CLOAD_SCRIPT="/usr/local/bin/python3.9 -m cfloader" make cload
```



## Build natively
In `external/crazyflie_firmware` run
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

# Installing cflib and cfclient

In your desired environment (e.g. venv)

```
pip3 install -e external/crazyflie_lib_python
pip3 install -e external/crazyflie_clients_python
```

# Running
In your desired environment (e.g. venv)
```
python3 -m cfclient.gui
```

To trigger the learned controller a modification to the cflib and the cfclient is needed. This is to receive the trigger signal from a button on the gamecontroller then format it into a cflib message that is sent to the crazyflie. In this way we can overwrite the normal controller as long as we hold a button and if the situation becomes unsafe we just release the button.f

The name of the added button is `learnedController` but it is not available in the configuration UI. Proceed by copying a good preset for the controller e.g. `PS3_Mode_1`. Then add/overwrite e.g. the exit button and save the config under a new name. In the console output of the cfclient.gui the path to the saved config should be shown. Edit it and change the `key` and `name` to e.g.:
```
{
    "id": 9,
    "scale": 1.0,
    "key": "learnedController",
    "name": "learnedController",
    "type": "Input.BUTTON"
}
```

Beware that there is a motor divider parameter that is set to a high value leading to low motor outputs by default. To "arm" the controller, change the parameter under the `bpt` group in the cfclient UI.

# Vicon
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



Extract checkpoints

```
find * | grep actor_000000000500000.h$ > /tmp/files
find * | grep actor_000000001500000.h$ >> /tmp/files
tar czvf checkpoints.tar.gz -T /tmp/files
```


```
cp ablation_data_organized/d+o+a+r+h+c+f+/000000000500000/0.h data/actor.h && make -j8 && CLOAD_CMDS="-w radio:
//0/80/2M" make cload
```