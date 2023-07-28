CRAZYFLIE_BASE := external/crazyflie_firmware

OOT_CONFIG := $(PWD)/config 
EXTRA_CFLAGS += -I$(PWD) -I$(PWD)/external/backprop_tools/include -std=c++17 -O3 -DBACKPROP_TOOLS_CONTROLLER

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
