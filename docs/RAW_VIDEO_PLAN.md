# Raw-video backbone plan (unvalidated)

A separate raw-video cache will make train-time features and inference-time features reproducible. Native author caches are useful for first baselines but cannot be assumed equivalent to this extraction route.

Initial candidates:
- Shared multilingual BERT contextual token features, pinned google-bert/bert-base-multilingual-cased revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8. Model card: Apache-2.0, Wikipedia pretraining. Cache all valid tokens, not only CLS.
- WavLM base-plus acoustic sequence, pinned microsoft/wavlm-base-plus revision 4c66d4806a428f2e922ccfa1a962776e232d487b. Pretrained primarily on English speech; Chinese utility is an empirical question, not an assumed capability. The card links the Microsoft UniSpeech license, which must be retained/checked separately before redistributing weights. Consider multilingual Whisper encoder/emotion2vec if Chinese validation indicates weakness.
- ImageNet pretrained ResNet18 sampled frame embeddings as an inexpensive visual starting point; evaluate the actual benefit versus stronger face/video representations. No identity inference or face-recognition task.
- Multilingual Whisper base for raw-audio transcription, with each clip language detected independently. Provided transcripts are a separate oracle-transcript condition; measure ASR degradation on validation rather than hiding the mismatch.

Record duration, audio receptive-field timestamps, video sample timestamps and ASR word timestamps. Learned fusion must honor masks and absence flags. Text token offsets can be linked to ASR word spans; do not infer physical cross-modal alignment from native author feature indices.

Do not replace final trained checkpoints with random weights in demos. A raw video is not successfully supported until extracted features from this same route pass a trained checkpoint and produce complete five-label predictions with measured runtime.
