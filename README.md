# Recommendation Letter Drafting Assistant

This Streamlit app uses two Hugging Face pipelines:

1. A fine-tuned `prajjwal1/bert-tiny` text-classification model that predicts recommendation strength from transcript-like student profiles.
2. `distilbert-base-uncased-finetuned-sst-2-english` sentiment analysis to check whether faculty notes are positive or require review.

The app generates an editable recommendation letter draft for faculty review. It supports single-profile drafting and batch CSV drafting.

## Public Links

- GitHub repository: https://github.com/SuperMarioGotze27/recletter-drafting-assistant
- Fine-tuned model: https://huggingface.co/SuperMarioGotze27/recletter-strength-bert-tiny
- Streamlit app: https://recletter-drafting-assistant-4fh5aqrlvddloyjgfxgcnv.streamlit.app/

## Streamlit Cloud Settings

- Repository: `SuperMarioGotze27/recletter-drafting-assistant`
- Branch: `main`
- Main file path: `app.py`
- Required files: `requirements.txt`, `sample_student_profiles.csv`, and the local `recletter_strength_model/` folder

## Local Run

```bash
streamlit run app.py
```

The folder `recletter_strength_model/` must stay beside `app.py`.

## Notes

The dataset is synthetic and privacy-safe. The application produces a draft for faculty review; it is not an automated final recommendation sender.
