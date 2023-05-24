CRAZYFLIE_BASE := ../crazyflie-firmware

OOT_CONFIG := $(PWD)/config 
EXTRA_CFLAGS += -I$(PWD) -I$(PWD)/../../backprop_tools_multirotor/include -std=c++17 -O3

include $(CRAZYFLIE_BASE)/tools/make/oot.mk