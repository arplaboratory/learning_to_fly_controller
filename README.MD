Install dependencies according to the [official docs](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/)

```
cd external/crazyflie_firmware
make cf2_defconfig
cd ../../
make
cfloader flash build/cf2.bin stm32-fw -w radio://0/80/2M
```