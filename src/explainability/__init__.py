"""Explainability for the frozen final models (Task 15).

- text.py: explain_text() - Integrated Gradients (T05) or exact linear contributions (TF-IDF models)
- image.py: explain_image() - occlusion sensitivity (SigLIP + classifier)
- multimodal.py: explain_multimodal() - unimodal comparison, train-mean modality ablation, agreement
- moderation.py: recommend() - deterministic ALLOW / WARNING / HUMAN_REVIEW design (no automatic block)
- explanation.py: explain_meme() / explain_text_input() - the structured object for the future UI

All outputs are MODEL ATTRIBUTION: they show what moved the model's score, not why content is harmful,
and not proof of causal reasoning. Settings: config/explainability.yaml.
Kept outside the checksum-frozen folders (src/preprocessing, src/models, src/features, src/evaluation).
"""
