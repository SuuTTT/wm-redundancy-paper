# Raw-data archive

Compact raw experimental data pulled from the fleet before box teardown.
Contains the small, high-value artifacts only (NOT checkpoints/replay/videos).

- `rawdata_4070.tgz` (box 44941373, 4070): VBN fingerprint eval logs
  (`tdmpc_glass/exp/vac/logs/*.log`), Dreamer per-episode learning curves
  (`dreamerv3/logdir/*/scores.jsonl`), and launch logs (`*.out`).

All *derived* numbers used in the paper are additionally in `../bet2_null_results.md`
and issue SuuTTT/tdmpc-glass#8. Note: several early Dreamer runs (cup, reacher-n1,
finger, pendulum-n1, cartpole, walker-n1) were pruned during the campaign to manage
disk; their harvested numbers survive in the ledger but their raw curves do not.
