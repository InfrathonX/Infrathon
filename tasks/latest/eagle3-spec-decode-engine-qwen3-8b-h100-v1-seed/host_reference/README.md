# Host-only references

These controls and ladder implementations support parity, oracle-negative, and headroom qualification. They never ship: neither image build context copies `host_reference/`. `greedy_kv_control` is a lossless static-KV baseline; `upstream_direct` invokes the pinned vendor checkout; `lossy_control` deliberately emits unverified tokens; `fp8_target_control` deliberately round-trips target linear weights through FP8; `ladder` contains the measured L1-L3 legitimate variants.
