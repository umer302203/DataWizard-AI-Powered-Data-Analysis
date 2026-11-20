# 🧙‍♂️ DataWizard: AI-Powered Data Analysis

> *Chat with your CSV files — get instant dataset summaries, pandas-powered exploration, and auto ML model evaluation through natural language.*

[![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.21-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Gradio](https://img.shields.io/badge/Gradio-6.0-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red?style=for-the-badge)](https://www.linkedin.com/in/rana-umer-05a9a9359/)

---

## 📌 Overview

**DataWizard** is an AI-powered data science assistant that lets non-technical users analyze CSV datasets using **plain natural language**. Built on **LangChain**, **OpenAI GPT**, **pandas**, **scikit-learn**, and a polished **Gradio dashboard**, it turns raw spreadsheets into conversational insights — no coding required.

Upload or select any CSV file — and the system will instantly:

1. **Discover** every available dataset in the working directory
2. **Summarize** its structure (columns, data types, shape)
3. **Explore** it using pandas methods (`head`, `describe`, `info`, `tail`)
4. **Auto-detect** whether it's a classification or regression task
5. **Train & evaluate** a Random Forest model — returning accuracy or R²/MSE
6. **Display** results in a clean Gradio UI with **live ReAct reasoning logs**

**Why it matters:** Data science is gated behind Python knowledge. Business owners, managers, and domain experts sit on goldmines of CSV data but can't extract insights. DataWizard democratizes analytics — one chat message at a time.

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| 💬 **Natural Language Interface** | Ask questions like *"Which dataset is for regression?"* |
| 📂 **Dataset Discovery** | Auto-lists every `.csv` file in the working directory |
| 📤 **Custom CSV Upload** | Drag & drop your own dataset directly in the UI |
| 🧠 **In-Memory Caching** | Loads DataFrames once into `DATAFRAME_CACHE` — no token waste |
| 📊 **Dataset Summaries** | Column names, data types, and shape per file |
| 🔧 **DataFrame Method Runner** | Runs `head`, `tail`, `describe`, `info` via `getattr` |
| 🤖 **Auto ML Detection** | Agent decides classification vs regression on its own |
| 📈 **Model Evaluation** | RandomForestClassifier (accuracy) / RandomForestRegressor (R² + MSE) |
| 🔁 **ReAct Agent Executor** | Full think → act → observe loop until final answer |
| 🔗 **LCEL Chain Demo** | Side-by-side comparison of `prompt \| llm \| parser` vs tool-using agent |
| 🧠 **Live ReAct Trace** | Watch the agent's reasoning + tool calls in real-time |
| ⚡ **Quick Action Buttons** | One-click prompts for common workflows |
| 🎨 **Polished Gradio UI** | Blue Soft theme, dataset preview, chat + logs |
| 🛡️ **Error Handling** | Graceful fallbacks for missing files, bad columns, API errors |

---

## 🧠 How It Works

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATAWIZARD — PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│   │  User Query     │    │  ChatPrompt     │    │  GPT-4o /       │       │
│   │  (Gradio UI)    │───▶│  Template       │───▶│  GPT-4o-mini    │       │
│   │  + CSV Upload   │    │  (system+user+  │    │  (OpenAI)       │       │
│   └─────────────────┘    │   scratchpad)   │    └────────┬────────┘       │
│                           └─────────────────┘             │                │
│                                                            │                │
│                                          ┌─────────────────▼──────────┐    │
│                                          │   ReAct Agent (raw)        │    │
│                                          │   Decides next tool call   │    │
│                                          └────────────────┬───────────┘    │
│                                                           │                │
│                                                           ▼                │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                   AgentExecutor (ReAct Loop)                     │     │
│   │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │     │
│   │   │ Think   │───▶│  Act    │───▶│ Observe │───▶│ Repeat  │       │     │
│   │   └─────────┘    └────┬────┘    └─────────┘    └─────────┘       │     │
│   │                       │                                          │     │
│   │                       ▼                                          │     │
│   │   ┌───────────────────────────────────────────────────────┐      │     │
│   │   │              LangChain Tools (@tool)                  │      │     │
│   │   │  • list_csv_files        • call_dataframe_method      │      │     │
│   │   │  • preload_datasets      • evaluate_classification... │      │     │
│   │   │  • get_dataset_summaries • evaluate_regression...     │      │     │
│   │   └───────────────────────┬───────────────────────────────┘      │     │
│   │                           │                                      │     │
│   │                           ▼                                      │     │
│   │   ┌───────────────────────────────────────────────────────┐      │     │
│   │   │   DATAFRAME_CACHE (module-level)  +  pandas + sklearn │      │     │
│   │   └───────────────────────────────────────────────────────┘      │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                           │                │
│                                                           ▼                │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │       Gradio UI renders: Chat reply + Live ReAct logs            │     │
│   │       (contextlib.redirect_stdout captures verbose=True)         │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow — Step by Step

| Step | Component | What It Does |
|:----:|-----------|--------------|
| **1** | **Gradio UI** | User types a question or uploads a CSV |
| **2** | **`list_csv_files()`** | Agent discovers all `.csv` files in working dir |
| **3** | **`preload_datasets()`** | Loads DataFrames into `DATAFRAME_CACHE` (once) |
| **4** | **`get_dataset_summaries()`** | Returns columns + dtypes for reasoning |
| **5** | **`call_dataframe_method()`** | Runs `head`, `describe`, `info` via `getattr` |
| **6** | **`evaluate_*_dataset()`** | Trains Random Forest, returns metrics |
| **7** | **`AgentExecutor`** | Feeds observations back until final answer |
| **8** | **Gradio rendering** | Chat bubble + ReAct logs panel updated |

### The 6 LangChain Tools

Each tool is a **plain Python function** wrapped with `@tool`. The docstring is what the LLM reads to decide when to call it:

```
1. list_csv_files()                        → Discover .csv files
2. preload_datasets(paths)                 → Load into DATAFRAME_CACHE
3. get_dataset_summaries(paths)            → Columns + dtypes metadata
4. call_dataframe_method(file, method)     → getattr(df, method)()
5. evaluate_classification_dataset(...)    → RandomForestClassifier + accuracy
6. evaluate_regression_dataset(...)        → RandomForestRegressor + R² + MSE
```

**Why this matters:** Clear docstrings + strict type annotations (`List[Dict[str, Any]]`) are what let the LLM reliably pick the right tool and parse the result.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM Orchestration** | LangChain 0.3.21 + `langchain-openai` |
| **Language Model** | OpenAI GPT-4o / GPT-4o-mini (via `init_chat_model`) |
| **Data Handling** | pandas 2.2.3 + numpy 2.2.4 |
| **Machine Learning** | scikit-learn 1.6.1 (RandomForest) |
| **Visualization** | matplotlib + seaborn (extensible) |
| **UI Framework** | Gradio 6.0 |
| **Language** | Python 3.12+ |
| **Agent Type** | `create_openai_tools_agent` + `AgentExecutor` (ReAct) |
| **Expression Language** | LCEL (`prompt \| llm \| StrOutputParser()`) |

---

## 📁 Project Structure

```
datawizard/
│
├── 📓 LLM-Powered Data Science LCEL.ipynb   # Main notebook
├── 📄 README.md                             # This file
│
├── data/
│   ├── 📊 regression-dataset.csv            # Sample regression data
│   └── 📊 classification-dataset.csv        # Sample classification data
│
└── 🔒 .gitignore
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- OpenAI API key (for local use) **OR** IBM Skills Network lab environment (pre-configured)
- Jupyter Notebook / JupyterLab

### Step 1: Clone the Repository

```bash
git clone https://github.com/umer302203/datawizard.git
cd datawizard
```

### Step 2: Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install langchain==0.3.21 langchain-openai==0.3.10 openai==1.68.2
pip install pandas==2.2.3 numpy==2.2.4 matplotlib==3.10.1 seaborn==0.13.2
pip install scikit-learn==1.6.1 gradio
```

### Step 4: Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

> ⚠️ **Skills Network users:** no key needed — the sandbox uses a proxy model (`gpt-6-luna`).

### Step 5: Launch the Notebook

```bash
jupyter notebook "LLM-Powered Data Science LCEL.ipynb"
```

Run **all cells top to bottom**. The Gradio app launches at `http://127.0.0.1:7860` (with a public `*.gradio.live` share link).

---

## 🖥️ Usage Guide

### 1. Select or Upload a Dataset
- Pick a default CSV from the **dropdown**, or
- Click **"Upload Your Own CSV File"** to drag & drop
- Metadata (rows, columns, dtypes) + **first 5 rows preview** appears instantly

### 2. Ask a Question
- Default prompt: *"What columns are in the dataset?"*
- Or try: *"Is this dataset for classification or regression?"*, *"Train and evaluate the ML model."*

### 3. Use Quick Actions
Three buttons for one-click workflows:

| Button | Action |
|--------|--------|
| **1️⃣ List & Summarize All CSVs** | Discovers + summarizes every dataset |
| **2️⃣ Classification vs Regression Check** | AI decides task type per dataset |
| **3️⃣ Train & Evaluate ML Models** | Trains Random Forest + returns metrics |

### 4. Watch the ReAct Trace
The right panel shows every **thought → tool call → observation** live — powered by `verbose=True` + `contextlib.redirect_stdout` capture.

---

## 📋 Requirements (`requirements.txt`)

```
langchain==0.3.21
langchain-openai==0.3.10
openai==1.68.2
pandas==2.2.3
numpy==2.2.4
matplotlib==3.10.1
seaborn==0.13.2
scikit-learn==1.6.1
gradio>=6.0.0
```

---

## 🎨 Sample Output

**User:** *"Analyze all datasets and tell me which is for classification vs regression."*

**Agent's ReAct Trace:**
```
> Entering new AgentExecutor chain...

Invoking: `list_csv_files` with `{}`
[Observation] ['classification-dataset.csv', 'regression-dataset.csv']

Invoking: `get_dataset_summaries` with `{'dataset_paths': [...]}`
[Observation] [
  {"file_name": "classification-dataset.csv",
   "column_names": ["feature1", "feature2", ..., "target"],
   "data_types": {"feature1": "float64", ..., "target": "object"}},
  {"file_name": "regression-dataset.csv",
   "column_names": ["x1", "x2", ..., "price"],
   "data_types": {"x1": "float64", ..., "price": "float64"}}
]

Invoking: `evaluate_classification_dataset` with `{'file_name': 'classification-dataset.csv', 'target_column': 'target'}`
[Observation] {"accuracy": 0.9523}

Invoking: `evaluate_regression_dataset` with `{'file_name': 'regression-dataset.csv', 'target_column': 'price'}`
[Observation] {"r2_score": 0.8741, "mean_squared_error": 2451.32}

> Finished chain.

**Final Answer:**
- **classification-dataset.csv** → Classification task (`target` column is categorical)
  - Accuracy: **95.23%**
- **regression-dataset.csv** → Regression task (`price` column is continuous)
  - R² Score: **0.8741**
  - Mean Squared Error: **2451.32**
```

---

## 🐛 Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `Saving to: 'regression-dataset.csv.3'` | File already exists — `wget` added suffix | Use `wget -O regression-dataset.csv <url>` |
| `Model 'gpt-6-luna' not found` | Local run without sandbox proxy | Switch to `gpt-4o-mini` in `init_chat_model` |
| `theme` argument not recognized | `theme` belongs to `gr.Blocks()`, not `launch()` | Use `gr.Blocks(theme=gr.themes.Soft())` |
| `Agent keeps calling list_csv_files` | Expected ReAct behavior — gathering context first | Normal; follow up with a specific prompt |
| `value_counts` method fails | `call_dataframe_method` supports no-arg methods only | Use `head` / `describe` / `info` |
| `Target column not found` | Wrong column name passed by agent | Check exact column names in dataset preview |
| `OpenAI authentication error` | `OPENAI_API_KEY` not set | `export OPENAI_API_KEY="sk-..."` |

---

## 🚢 Deployment (Hugging Face Spaces or Render)

### Option 1: Hugging Face Spaces (Gradio SDK)

1. Create the Space with **Gradio** SDK
2. Upload `app.py` (converted from notebook) + `requirements.txt`
3. Add `OPENAI_API_KEY` to **Repository Secrets**
4. The app auto-launches

### Option 2: Streamlit Cloud / Render / Railway

1. Convert notebook → `app.py` (extract Gradio blocks)
2. Add a `Procfile`:
   ```
   web: python app.py
   ```
3. Deploy and set env vars in the dashboard

---

## 🔮 Future Improvements

- [ ] **Multi-file Analysis** — Compare multiple datasets side-by-side
- [ ] **SQL Support** — Connect to databases, not just CSVs
- [ ] **Auto Visualization** — Generate charts (histograms, correlations) on request
- [ ] **More Models** — XGBoost, LightGBM, Logistic Regression comparisons
- [ ] **Conversation Memory** — Multi-turn context across queries
- [ ] **Export Reports** — Download analysis as PDF / HTML
- [ ] **RAG over Data Dictionaries** — Understand column semantics automatically
- [ ] **Multilingual Support** — Urdu, Hindi, Arabic responses

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

This project is distributed under the **MIT License** — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **LangChain** — For the tools + ReAct agent framework
- **OpenAI** — For GPT-4o multimodal reasoning
- **pandas team** — For the data manipulation backbone
- **scikit-learn** — For battle-tested ML models
- **Gradio** — For the beautiful UI framework

---

## 📬 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rana-umer-05a9a9359/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/umer302203)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Umer78786)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&text=DataWizard&fontSize=28&fontColor=white&fontAlignY=65" />
</p>

> Built with ❤️ by [Rana Umer](https://www.linkedin.com/in/rana-umer-05a9a9359/) 🚀
```

---

## ✅ Kya-kya match hua style ke saath:

| Element | AI Nutrition Coach | DataWizard (New) |
|---|---|---|
| Top hero section with tagline | ✅ | ✅ |
| Badges row (Python, framework, AI, license) | ✅ | ✅ |
| 📌 Overview with numbered list | ✅ | ✅ |
| 🎯 Features table | ✅ | ✅ |
| 🧠 Architecture ASCII diagram | ✅ | ✅ (bigger, ReAct loop included) |
| Data Flow step table | ✅ | ✅ |
| 🛠️ Tech Stack table | ✅ | ✅ |
| 📁 Project Structure | ✅ | ✅ |
| 🚀 Installation with steps | ✅ | ✅ |
| 🖥️ Usage Guide | ✅ | ✅ |
| 📋 Requirements | ✅ | ✅ |
| 🎨 Sample Output | ✅ | ✅ (ReAct trace + final answer) |
| 🐛 Troubleshooting | ✅ | ✅ (6 specific DataWizard bugs) |
| 🚢 Deployment | ✅ | ✅ |
| 🔮 Future Improvements (checklist) | ✅ | ✅ |
| 🤝 Contributing | ✅ | ✅ |
| 📄 License | ✅ | ✅ |
| 🙏 Acknowledgments | ✅ | ✅ |
| 📬 Connect with Me | ✅ | ✅ (same links) |
| Footer wave + signature | ✅ | ✅ |

Bas `README.md` mein paste karo aur GitHub pe push kar do. 🎉 Agar chaho to main ismein **screenshots section** bhi add kar deta hoon (jahan tum Gradio UI ka screenshot daal sako) — bata dena.
