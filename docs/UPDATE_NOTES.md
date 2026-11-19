# Pure-weight release update (2026-11-19)

The public batch evaluator now loads M1_weights.pt through M5_weights.pt from
their existing numbered model directories. It does not fall back to older
best_patient_mean_macro_f1.pt files. A flat directory is supported explicitly
with --weights_layout flat. Each direct Drive link and observed file size was
updated, and all five files had public reader permissions at the check time.

Pure parameters need no private training config or experiments directory.
The repository publishes its inference settings, resolves the five known
filenames to their architectures, and rejects conflicting model identities.
A renamed bare dictionary needs --model_kind. M4/M5 are not distinguished using
shape guesses. Trained tensor keys/shapes are restored strictly. The evaluator
uses weights_only=True and rejects non-tensor model-state values.

BERT still needs its architecture config and matching tokenizer vocabulary.
Inference constructs ResNet without pretrained weights and restores the trained
ResNet and BERT parameters from the supplied pure weight dictionary.
No epoch is inferred for pure weights; its metrics epoch is null.

The original data/sample.csv and all data/sample_images blobs remain unchanged.
The five architecture definitions, transformations, losses and training loop
are unchanged. Existing full research checkpoints remain supported by the
single-model entry when explicitly selected.

The demonstration does not establish independent validation or lack of training
overlap. Pure-weight metadata removal changes neither training provenance nor
predictions when every parameter and inference setting is the same.

Validation covers filename resolution, fixed defaults, numbered/flat commands,
manifest consistency, missing-file preflight, bare-weight mocked evaluation,
legacy checkpoint handling and existing data/metric workflows. The optional
real-model smoke test is skipped if model libraries are unavailable. Large
Drive weight tensors and real photograph inference were not tested here.
