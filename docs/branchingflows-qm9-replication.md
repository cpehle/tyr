# BranchingFlows QM9 Molecule Generation Replication

This note records how to reproduce the molecule generation example described by
the Branching Flows paper in this Tyr checkout.

## Sources Checked

- `../BranchingFlows.jl`: contains the generic Julia `CoalescentFlow`,
  `branching_bridge`, and forward-time `Flowfusion.step`/`gen` pattern, plus a
  toy mixed continuous/discrete demo. It does not contain a QM9 molecule example.
- `../MoleculeFlow.jl`: RDKit/PythonCall wrapper for molecule IO, fingerprints,
  descriptors, drawing, and analysis. It is useful for evaluation/export, not as
  the generator implementation.
- arXiv source for `2511.09465`: the QM9 recipe is in
  `refactored_appendices/F2_analyses.tex`.

## Published QM9 Setup

Data:

- Use QM9 coordinate data.
- Keep the canonical-SMILES heavy atom order.
- Move hydrogens so each hydrogen is inserted before its nearest heavy atom,
  ordered by distance if multiple hydrogens attach to the same heavy atom.
- Each sequence element is one atom: 3D coordinates plus a discrete atom label.

Branching flow:

- Initial state length is always 1.
- Initial coordinates are sampled from `N(0, 1)` per coordinate.
- Initial discrete label is the mask/dummy token.
- Continuous base process: endpoint-conditioned time-inhomogeneous OU bridge
  with mean reversion `theta = 5`; variance decays from `10` at `t = 0` to
  `0.001` by `t = 1`.
- Discrete base process: DFM convex interpolation between `x0`, uniform noise,
  and `x1`, using `Beta(2, 2)` for both hazards and `omega_u = 0.2`.
- Branching hazard distribution: `Beta(1, 3/2)`.
- Deletion hazard distribution: `Uniform(0, 1)`.
- Deletion padding: sample a Poisson number of duplicate-to-delete atoms with
  mean `20%` of the `x1` length, with replacement, inserting duplicates on
  either side of the original. In the Julia/Tyr bridge parameterization this is
  `deletionPad := 1.2`; the value is the expected padded-length multiplier
  when it is at least `1.0`, not the extra fraction.
- Anchors: weighted average for continuous coordinates, mask token for discrete
  atom labels.

Model/training:

- 12-layer transformer.
- 12 heads, head dimension 64, embedding dimension 384.
- Atom positions use Random Fourier Features.
- Pairwise spatial features feed each layer through learnable layer/head-specific
  attention bias.
- RoPE encodes primary sequence order.
- Final six layers additively update atom positions and recompute spatial pair
  features.
- Heads predict endpoint coordinates, endpoint atom-label logits, log expected
  future bifurcations by `t = 1`, and deletion logits.
- The appendix says 500k batches of size 128 with Muon, post-warmup LR `0.005`,
  and linear cooldown for the last 50k batches. A later ablation section says the
  main unconditional figure models used 800k iterations, so parity target should
  be chosen explicitly.
- Sampling uses the schedule
  `t = 1 - (cos(pi * s) + 1) / 2` for `s = 0, 1/1000, ..., 1`.

Evaluation:

- Export atom point clouds to `.xyz`.
- Use OpenBabel to infer bonds and convert `.xyz` to `.sdf`.
- Use RDKit, locally available through `../MoleculeFlow.jl`, for SMILES,
  descriptors, fingerprints, validity, uniqueness, and UMAP inputs.
- PoseBusters metrics are used in the appendix, in addition to distributional
  comparisons of atom counts and molecular descriptors.

## Tyr Mapping

Already close:

- `Tyr/Model/BranchingFlows.lean` has forest sampling, bridge sampling,
  group minima, deletion padding, coalescence policies, and generic anchor
  merging.
- It also now has Julia-compatible time distributions for `Uniform(0, 1)`,
  `Beta(1, 2)`, `Beta(1, 3/2)`, and `Beta(2, 2)`, the default split-intensity
  transform, and scalar/array loss helpers for split counts and deletion logits.
- `Tyr/Model/BranchingFlowsTrain.lean` has discrete and continuous batch packers
  and losses.
- `Examples/BranchingFlows/MixedTrainDemo.lean` is the local toy analogue of the
  Julia mixed continuous/discrete demo.
- `Tyr/Model/BranchingFlows/DiffEq.lean` makes Tyr DiffEq solvers usable as
  BranchingFlows bridges for ODE/SDE prototypes.
- It also contains a scalar endpoint-conditioned OU bridge with constant or
  log-linear variance schedules. This covers the continuous base-process formula
  needed for one coordinate channel and can be lifted over 3D atom positions.
- `Tyr/Model/BranchingFlows/Discrete.lean` ports the Flowfusion
  `DistNoisyInterpolatingDiscreteFlow` schedule: target, uniform, and
  source/mask mixture weights; conditional bridge weights; categorical bridge
  sampling; Euler-style label stepping; and the label-loss time scaling.
- `Tyr/Model/BranchingFlows/Molecule.lean` adds a molecule-shaped atom record:
  3D coordinates plus a discrete label, QM9-style OU defaults, a mask-token
  anchor merge, and a bridge that runs OU coordinates plus optional DFM labels
  through `branchingBridge`.
- `Tyr/Model/BranchingFlows/MoleculeTrain.lean` packs molecule bridge results
  into one tensor batch with coordinates, coordinate anchors, labels,
  label-anchor targets, DFM label-loss scales, masks, split targets, and
  deletion targets.  `moleculeLosses` consumes those scales, so
  `packBranchingMoleculeWithDFM` applies the Flowfusion label-loss time factor.
- It also exposes `BranchingMoleculeModel` and `trainStepMolecule`, the
  QM9-shaped autograd/AdamW training step over coordinate, label, split, and
  deletion heads.
- `Tyr/Model/BranchingFlows/Molecule.lean` also has `MoleculeModelPrediction`,
  `moleculeBranchingStep`, and `moleculeBranchingGenerate`, which adapt
  coordinate endpoint predictions plus atom-label logits into the generic
  forward event path using OU coordinate stepping and DFM label stepping.
- `Tyr/Model/BranchingFlows/QM9.lean` defines the preprocessing boundary for
  QM9 records.  It parses single JSON molecules, JSON molecule arrays, and
  JSONL batches; validates atom count, finite coordinates, label vocabulary,
  and reserved mask-token use; and converts records into
  `BranchingState MoleculeAtom` plus the length-one masked source state used by
  generation.
- It also contains the molecule-specific deletion-padding hook
  `maskDeletedMoleculeLabels` and raw `.xyz` export helpers
  `moleculeStateToXYZ`/`writeMoleculeXYZ`, including optional token-to-symbol
  mapping for mask labels.
- `Examples/BranchingFlows/MoleculeGenerationDemo.lean` is the runnable local
  smoke example. It parses an inline preprocessed QM9-style JSONL fixture, runs
  a bridge sample with `deletionPad := 1.2`, applies the deleted-label mask
  modifier, runs `moleculeBranchingGenerate` with an oracle prediction model,
  and writes target, bridge, and generated `.xyz` files.
- `Examples/BranchingFlows/MoleculeTrainDemo.lean` is the runnable training
  smoke. It parses an inline preprocessed molecule fixture, builds a
  QM9-shaped bridge batch, trains a small coordinate/label model through
  `trainStepMolecule`, and fails if total, coordinate, or label loss does not
  decrease.
- `Tyr/Model/BranchingFlows/MoleculeTransformer.lean` adds a reusable
  Torch-backed molecule transformer with head-specific pairwise
  coordinate-distance attention bias and endpoint-coordinate, atom-label,
  split-count, and deletion heads.
- `Examples/BranchingFlows/MoleculeTransformerTrainDemo.lean` trains that
  transformer on the same QM9-shaped bridge path, with nonzero split and
  deletion loss weights, and fails if any of the four head losses does not
  decrease.

Preprocessed molecule records should look like:

```json
{"name":"example","smiles":"O","atoms":[{"label":8,"coord":[0.0,0.1,-0.2]}]}
```

Atom coordinates may also be emitted as `x`, `y`, and `z` fields.  The label
field should be the integer atom-token id used by the model vocabulary; the
loader also accepts `atom_label` as an alias.

Run the local smoke example with:

```bash
lake env lean --run Examples/BranchingFlows/MoleculeGenerationDemo.lean /tmp/tyr_branching_molecule_demo
```

It writes:

- `/tmp/tyr_branching_molecule_demo_target.xyz`
- `/tmp/tyr_branching_molecule_demo_bridge.xyz`
- `/tmp/tyr_branching_molecule_demo_generated.xyz`

The generated file should contain a three-atom water-shaped sample from the
oracle model. This is a shape and event-path check, not a trained molecular
model.

Convert QM9 `.xyz` coordinate files into Tyr's JSONL schema with:

```bash
python3 scripts/qm9_xyz_to_branching_jsonl.py /path/to/qm9_xyz_dir --out data/qm9_branching.jsonl
```

For a repository-local fixture smoke:

```bash
python3 scripts/qm9_xyz_to_branching_jsonl.py Examples/BranchingFlows/qm9_xyz --out /tmp/tyr_qm9_fixture.jsonl
```

Run the local molecule training smoke with:

```bash
lake exe BranchingFlowsMoleculeTrain
```

This is a trainability check, not the paper-scale architecture. It should print
initial/final total, coordinate, and label losses and fail if the tiny model
does not overfit the fixed molecule fixture.

Run the transformer training smoke with:

```bash
lake exe BranchingFlowsMoleculeTransformerTrain
```

This is still a small one-block overfit check, but it proves that the
QM9-shaped transformer, pairwise spatial attention bias, and all four molecule
heads train through `trainStepMolecule`.

Missing for paper-faithful molecule generation:

- Full-dataset QM9 preprocessing validation against QM9PACK/RDKit metadata. The
  repository-local `scripts/qm9_xyz_to_branching_jsonl.py` covers coordinate
  parsing, heavy-atom order preservation, and nearest-heavy hydrogen insertion.
- Bridge-variance or stochastic-coordinate training targets if stochastic OU
  bridge samples are used rather than deterministic conditional means.
- Scaling the molecule transformer from the local one-block overfit smoke to
  the paper-scale 12-layer architecture with random Fourier position features,
  RoPE sequence encoding, additive coordinate updates in the final layers, and
  the published optimizer/schedule.
- OpenBabel/RDKit evaluation scripts for generated samples.

## Practical Replication Order

1. Run `Examples/BranchingFlows/MoleculeGenerationDemo.lean` to validate the
   local molecule state shape, bridge sampler, deletion-padding mask hook,
   molecule forward sampler, and `.xyz` export path.
2. Use `moleculeBranchingGenerate` for real molecule sampling. Its model sees
   both `s1` and `s2`, so it can convert endpoint coordinate predictions and
   atom-label logits into OU/DFM stepping over each schedule interval.
3. Run `scripts/qm9_xyz_to_branching_jsonl.py` on QM9 `.xyz` files to emit the
   `QM9.lean` JSON/JSONL schema. Test on the tiny fixture batch before training.
4. Use `lake exe BranchingFlowsMoleculeTrain` to validate the molecule training
   path on a fixed overfit target.
5. Use `lake exe BranchingFlowsMoleculeTransformerTrain` to validate a
   transformer model with pairwise spatial attention bias and all molecule
   heads on a fixed overfit target, then scale to QM9.
6. Export generated `.xyz`, convert with OpenBabel, evaluate via
   `../MoleculeFlow.jl`/RDKit.

The shortest useful local demo is step 1. The shortest paper-faithful QM9
replication needs the full sequence above plus a substantial GPU training run.
