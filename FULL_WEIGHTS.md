# Full weights · self-contained

`teacher_shards/` now holds the complete 2.01 GiB of 219 teacher COPY tensors (22 upstream shards, each LIVE git-blob-sha1 verified, split into 32 MiB subshards). Together with `weights_v10/` (312 SGC/sidecar NEW tensors) and `student_smm_v10_3.safetensors` (48 SMM NEW tensors), this repo is **self-contained**: no external reconstruction is required.

Run `python materialize_full_student_v10_3.py --out student_full.safetensors` on a machine with ~2.6 GiB free disk. It reassembles all 579 tensors and recomputes sha256 for every entry before writing.

See `FINAL_MIGRATION_PROOF.md` for the 7-group proof that the migration is real.
