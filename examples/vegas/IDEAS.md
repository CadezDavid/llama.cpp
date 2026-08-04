# Private Vegas ideas

## Dynamic gamma (tested, not retained)

Two variants were tested on Qwen35 q8_0/turbo4 and removed. Acceptance-guided
gamma averaged 2.84 and reached 90.1 tok/s. An eight-cycle throughput search
settled on gamma 2 but reached only 88.1 tok/s. Fixed gamma 2 reached 127.8
tok/s on the same 62K conversation.

The main problems are exploration cost in 128-token replies and
gamma-dependent numerical trajectories. Revisit only if calibration can be
performed without committing exploratory output, or if generation is long
enough to amortize a shadow calibration phase.
