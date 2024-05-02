Install dependencies according to the [official docs](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/)

```
cd external/crazyflie_firmware
make cf2_defconfig
cd ../../
make
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```


### macOS: Building using Docker
```
./docker_build.sh
```
The build process needs access to the git repo (to incorporate the commit/tag into the firmware I suppose) hence the topmost repository (conttaining the `.git` folder) has to be mounted in the docker container when this repo is cloned as a submodule:
```
(cd .. && ./controller/docker_run.sh )
```