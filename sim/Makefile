# cocotb + Verilator. Defaults run the Phase 0 counter test.
#   make MODULE=test_counter TOPLEVEL=counter VERILOG_SOURCES="$PWD/../rtl/spike/counter.sv"
SIM           ?= verilator
TOPLEVEL_LANG ?= verilog

VERILOG_SOURCES ?= $(abspath ../rtl/spike/counter.sv)
TOPLEVEL        ?= counter
MODULE          ?= test_counter

# Waveforms land in sim_build/dump.fst (open with gtkwave). CI passes WAVES=0.
WAVES ?= 1

include $(shell cocotb-config --makefiles)/Makefile.sim
