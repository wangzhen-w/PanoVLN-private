# UniK3D Runtime Source

This directory contains the UniK3D 0.1 runtime package from
https://github.com/lpiccinelli-eth/UniK3D. Dataset and demo assets are
not included.

UniK3D is distributed under CC BY-NC-SA 4.0; see `LICENSE`.
Model checkpoints are not stored in this repository.

Runtime adaptations avoid importing optional benchmark/KNN dependencies from
the utility package initializer. `configs/large.json` stores only the model
architecture needed to rebuild UniK3D-Large from an embedded state dict.
The decoder exposes its three-stage ray-conditioned feature pyramid for VLN
multiscale geometry fusion.
