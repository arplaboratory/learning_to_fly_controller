The CrazyFlie firmware is expected in `../crazyflie-firmware`

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