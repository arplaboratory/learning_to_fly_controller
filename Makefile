CRAZYFLIE_BASE := ../crazyflie-firmware

#
# We override the default OOT_CONFIG here, we could also name our config
# to oot-config and that would be the default.
#
OOT_CONFIG := $(PWD)/config

include $(CRAZYFLIE_BASE)/tools/make/oot.mk