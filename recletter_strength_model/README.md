# RecLetter Strength BERT Tiny

This folder contains the local model files loaded by `app.py` on Streamlit Cloud.

The public fine-tuned model is also available at:

https://huggingface.co/SuperMarioGotze27/recletter-strength-bert-tiny

The model is a fine-tuned `prajjwal1/bert-tiny` text classifier for three labels: `Moderate Recommendation`, `Strong Recommendation`, and `Exceptional Recommendation`.

The training data is synthetic and privacy-safe. High held-out accuracy mainly shows that the model learned the designed synthetic recommendation-strength rules, not real-world generalization.
