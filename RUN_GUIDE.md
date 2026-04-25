# Step-by-Step Execution Guide: LLM Observability Framework

This guide walks you through running the entire observability framework using Google Colab. The process is broken down into four sequential notebooks.

## Prerequisites
- A Google Account (for Google Colab).
- Your GitHub repository link: `https://github.com/AniketAwchare/iitp_project_mk1.git`

---

## Process 1: Infrastructure Setup (Notebook 01)
**Goal:** Install dependencies, load financial datasets, build the RAG vector index, and load the Mistral 7B LLM into GPU memory.

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File** > **Open notebook** > **GitHub** tab.
3. Paste your repository URL: `https://github.com/AniketAwchare/iitp_project_mk1`
4. Select `notebooks/colab_01_infrastructure.py`.
5. **CRITICAL:** Before running, go to **Runtime** > **Change runtime type** > Select **T4 GPU** > Save.
6. Run the cells one by one (click the "Play" button or press Shift+Enter).

⏳ **Estimated Time:** 
- Dependency Installation: ~3-4 minutes
- Dataset Loading & FAISS Index Build: ~1-2 minutes
- Loading Mistral 7B model: ~3-4 minutes
- **Total Time:** ~8-10 minutes

---

## Process 2: Modules Demo (Notebook 02)
**Goal:** Test the four observability modules (Consistency, Drift, Alignment, Anomaly) against the live LLM.

1. Ensure Notebook 01 is fully executed and the LLM is loaded in memory.
2. Since Colab wipes memory if you switch tabs, the easiest way to run this is to **copy the cells from `colab_02_modules_demo.py` and paste them at the bottom of your active Notebook 01 session**, OR run the script directly via `!python notebooks/colab_02_modules_demo.py` if you cloned the repo in Colab.
3. Run the cells sequentially. You will see live outputs of consistency scores, drift detection, RAG alignment, and anomaly detection.

⏳ **Estimated Time:** 
- Running all modules on test queries: ~5 minutes
- **Total Time:** ~5 minutes

---

## Process 3: Run Experiments (Notebook 03)
**Goal:** Execute the three core experiments (Domain Shift, Adversarial Prompts, RAG Failure) and generate statistical results and charts.

1. This process does *not* require the heavy LLM to be loaded, as it uses smaller embedding models and synthetic data.
2. You can run this in a new Colab session or continue in the existing one.
3. Open `notebooks/colab_03_experiments.py` via GitHub.
4. Run all cells. It will evaluate thousands of queries and save the results to the `./results/` folder.

⏳ **Estimated Time:** 
- Domain Shift Experiment: ~1 minute
- Adversarial Experiment: ~1 minute
- RAG Failure Experiment: ~1 minute
- **Total Time:** ~3-4 minutes

---

## Process 4: Launch Dashboard (Notebook 04)
**Goal:** Start the Streamlit web dashboard to visualize the real-time metrics and experiment results.

1. Open `notebooks/colab_04_dashboard.py` in Colab.
2. Run the cells. It will start a local web server and use `pyngrok` (or `localtunnel`) to create a public link.
3. Click the generated public link (e.g., `https://xxxx.ngrok-free.app`) to view your interactive dashboard.

⏳ **Estimated Time:** 
- Launching Server & Tunnel: ~1-2 minutes
- **Total Time:** ~1-2 minutes

---

## Troubleshooting & Fixing Common Errors

### 1. `CUDA out of memory` Error
- **Cause:** The T4 GPU (15GB VRAM) ran out of memory while loading the Mistral 7B model.
- **Fix:** 
  - Ensure you have selected **T4 GPU** in the Runtime settings.
  - Restart the Colab session (**Runtime** > **Restart session**). Do not run multiple models or heavy tasks before Notebook 01.
  - The framework is pre-configured to use 4-bit quantization (`model_quantize_4bit = True` in config), which is required to fit 7B parameters on a T4.

### 2. `NameError: name 'pipeline' is not defined`
- **Cause:** You restarted the Colab runtime between running Notebook 01 and Notebook 02, wiping the loaded model from memory.
- **Fix:** You must run the module demo in the *same active session* where the model was loaded. Run Notebook 01 first, then execute the Notebook 02 code in that same environment.

### 3. `ModuleNotFoundError: No module named 'xyz'`
- **Cause:** The pip installation step failed, or you started a fresh session without installing dependencies.
- **Fix:** Run the dependency installation cell (Cell 1 in Notebook 01) again. Wait for it to finish successfully before moving on.

### 4. HuggingFace `401 Client Error` (Authentication)
- **Cause:** HuggingFace sometimes requires authentication or acceptance of terms to download models like Mistral.
- **Fix:** 
  - Create a free account at HuggingFace and accept the terms for the Mistral model.
  - Generate an Access Token in your HF settings.
  - In Colab, add and run this code before loading the model:
    ```python
    from huggingface_hub import login
    login("YOUR_HF_TOKEN")
    ```

### 5. Streamlit Dashboard Link Not Loading (ngrok error)
- **Cause:** `pyngrok` limits anonymous tunnels, or the tunnel timed out.
- **Fix:**
  - Create a free account at [ngrok.com](https://ngrok.com/) and copy your Auth Token.
  - Update Notebook 04 to include your token: `ngrok.set_auth_token("YOUR_TOKEN")` before calling `ngrok.connect(8501)`.
  - Alternatively, use Option B (`localtunnel`) provided in Notebook 04.

### 6. Missing Result Files / Blank Charts in Dashboard
- **Cause:** The dashboard is looking for CSV files in the `./results/` directory, but they haven't been generated yet.
- **Fix:** You must fully run Notebook 03 (`colab_03_experiments.py`) to generate the experiment data before the dashboard can display those charts. Ensure you run the dashboard in the same Colab session where the results were saved.
