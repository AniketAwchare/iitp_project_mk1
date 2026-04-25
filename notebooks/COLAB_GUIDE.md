# How to Run These Notebooks in Google Colab
## LLM Observability Framework — Financial Services

---

## Option 1: Upload & Open in Colab (Quickest)

1. Upload the entire `llm-observability-framework/` folder to your **Google Drive**
2. Open [colab.research.google.com](https://colab.research.google.com)
3. Go to **File → Open notebook → Google Drive** and open any `notebooks/colab_0X_*.py`
4. Colab will treat it as a script — run cells with the `# %%` markers

## Option 2: GitHub → Colab

1. Push this repo to GitHub
2. In Colab: **File → Open notebook → GitHub** → paste your repo URL
3. Open the `.py` files directly

## Option 3: Clone in Colab (Recommended for full run)

Paste this in the first cell of a new Colab notebook:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Or clone from GitHub
!git clone https://github.com/<your-username>/llm-observability-framework.git
%cd llm-observability-framework
!pip install -r requirements.txt
```

---

## Notebook Overview

| File | What it does | GPU needed? |
|---|---|---|
| `colab_01_infrastructure.py` | Install deps, load datasets, build FAISS index, load Mistral 7B | ✅ Yes (T4/A100) |
| `colab_02_modules_demo.py`   | Demo all 4 modules with real queries                            | ✅ Yes (T4) |
| `colab_03_experiments.py`    | Run all 3 experiments, produce result tables                   | ❌ CPU OK |
| `colab_04_dashboard.py`      | Launch Streamlit via ngrok tunnel                              | ❌ CPU OK |

---

## GPU Selection in Colab

**Runtime → Change runtime type → T4 GPU** (free)

For Mistral 7B (4-bit), T4 (15GB) is sufficient.  
If T4 is unavailable, use **Kaggle** (P100, 30hr/week free).

---

## Kaggle Alternative

1. Go to [kaggle.com/code](https://kaggle.com/code) → New Notebook
2. Settings → Accelerator → **GPU P100**
3. In the first cell:
```python
!git clone https://github.com/<your-username>/llm-observability-framework.git
%cd llm-observability-framework
!pip install -r requirements.txt -q
```
Then run the `colab_0X_*.py` scripts.
