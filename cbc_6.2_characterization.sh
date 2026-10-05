#!/bin/sh
# CBC (paper Alg. 1), Saturn. Build with tx_controller = 'cbc'.
# --force-max-simd-bitwidth=64 selects the ice scalar Tx path, where
# rte_eth_tx_done_cleanup() is supported.
sudo python3 ./run_ubenchmark.py \
	--app /home/ubuntu/dpdk/x86_64-native-linuxapp-gcc/examples/dpdk-tx_shaper_baseline \
	--eal-args=--force-max-simd-bitwidth=64 \
	--mechanism cbc \
	--lcore-sets '1,2' \
	--repeats 30 \
	--rates 3125000000 \
	--transient-types 0 \
	--cbc-poll-us 10 \
	--descs 4096 \
	--bursts 512 \
	--samples 500000 \
	--output 6.2_cbc_characterization
