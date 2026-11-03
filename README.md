# NLI MiniLM Colab (CS2308 Lab 2)

Fine-tune `microsoft/MiniLM-L12-H384-uncased` (~33M params, under 40M) for English NLI (MNLI + SNLI), then export `submission.zip` = `model.py` + `MODEL/`.

## Google Colab (recommended)

1. Open [`train_colab.ipynb`](./train_colab.ipynb) in Colab (**Runtime → GPU**).
2. Run all cells: clone this repo → install deps → train → download `submission.zip`.

Or open directly:

```text
https://colab.research.google.com/github/<USER>/nli-minilm-colab/blob/main/train_colab.ipynb
```

## Local

```bash
pip install -r requirements.txt
python train.py --epochs 2
```

## Submit to CodaBench

Upload `submission.zip` (contains `model.py` and `MODEL/`). Upload best `model.py` to Moodle separately.
