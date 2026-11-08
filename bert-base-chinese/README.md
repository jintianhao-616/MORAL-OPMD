# Local Chinese BERT files

The current five-model scripts default to the explicit Hugging Face model ID
`google-bert/bert-base-chinese`. This avoids confusing this placeholder directory
with a complete local model.

For offline training, place the downloaded BERT `config.json`, `vocab.txt`,
tokenizer files and model weights in this directory, then pass
`--bert_path ./bert-base-chinese`.
For evaluation, the checkpoint restores BERT parameters; local tokenizer files
and `config.json` are still required. ResNet50 weights are also restored from the
checkpoint during evaluation. Offline training additionally needs cached
ImageNet ResNet50 weights.

No pretrained BERT weight file is bundled here.
