# LLMForecaster — Agentic Time Series Forecasting with sktime

> **LLM as planner. sktime as executor. Validation gate as safety layer.**

A working prototype of an agentic forecasting system built for the [sktime ESoC 2026 agentic track](https://github.com/sktime/sktime/issues/9721). The LLM selects and configures a sktime estimator from natural language — all statistical computation stays inside sktime.

---

## 🎯 The Core Idea

Most LLM + forecasting approaches fall into two traps:
- **Direct LLM forecasting** — LLM generates numbers directly. Non-deterministic, unvalidatable, unreliable.
- **Heavy orchestration** — LangChain, LlamaIndex, RAG pipelines. Complex, hard to test, opaque.

This prototype takes a different path: **the LLM is only a planner.**

```
Natural language prompt
        ↓
   [LLM Backend]          ← any callable: Groq, Gemini, OpenAI, mock
        ↓
  Estimator + params      ← structured JSON output
        ↓
 [Validation Gate]        ← fit on small slice, catch hallucinations
        ↓
  sktime forecaster       ← real statistical computation
        ↓
     Forecast
```

---

## 🔑 Key Innovation: Validation-Gated Selection

Before committing to the LLM's choice, the system runs a lightweight sanity check:

1. LLM returns `{"estimator": "AutoARIMA", "params": {...}, "reasoning": "..."}`
2. System attempts instantiation + fit on a small data slice
3. ✅ If it works → fit on full data, proceed
4. ⚠️ If it fails → fallback estimator used transparently, user notified

This catches LLM hallucinations **before they reach the user** — using sktime itself as the validator. No external tooling needed.

---

## 📓 Notebooks

### 1. `sktime_llm_forecaster_groq.ipynb` — Real LLM Demo
Uses **Groq API (Llama-3.1-8b-instant)** as the LLM backend. Demonstrates:
- Real LLM selecting sktime estimators from natural language
- Validation gate catching bad parameter suggestions
- Graceful fallback when validation fails
- Three different prompts → three different model selections

**Requires:** Groq API key (free at [console.groq.com](https://console.groq.com))

### 2. `sktime_mcp_agentic_demo.ipynb` — Mock LLM Demo
Uses a deterministic `RuleBasedLLM` — no API key needed. Demonstrates:
- Full architecture working offline and reproducibly
- Validation gate catching hallucinated estimator names
- Pluggable backend interface — swap backends with zero code changes
- CI-safe: no network calls, no API keys

**Requires:** Nothing beyond sktime

---

## 🚀 Quick Start

```bash
pip install sktime groq matplotlib jupyter
```

```python
from groq import Groq
import json

client = Groq(api_key="your-groq-key")

def groq_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# Plug in any LLM backend
forecaster = LLMForecaster(
    llm=groq_llm,
    prompt="Use a seasonal model for monthly data with strong trend",
    fallback="NaiveForecaster",
)

forecaster.fit(y_train)
y_pred = forecaster.predict(fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# Inspect what the LLM chose
print(forecaster.selected_estimator_)   # e.g. "AutoETS"
print(forecaster.selection_reasoning_)  # LLM's explanation
print(forecaster.used_fallback_)        # True if fallback was triggered
```

---

## 🔌 Pluggable LLM Backends

The `llm` parameter accepts **any callable** with signature `(prompt: str) -> str`:

```python
# Groq (free, fast)
def groq_llm(prompt): ...

# Gemini
def gemini_llm(prompt):
    return model.generate_content(prompt).text

# OpenAI
def openai_llm(prompt):
    return client.chat.completions.create(...).choices[0].message.content

# Mock (no API key — for testing and CI)
def mock_llm(prompt):
    return json.dumps({"estimator": "NaiveForecaster", "params": {}, "reasoning": "test"})

# Rule-based (deterministic, offline)
def rule_llm(prompt):
    if "seasonal" in prompt.lower():
        return json.dumps({"estimator": "ExponentialSmoothing", "params": {}, "reasoning": "seasonal"})
    return json.dumps({"estimator": "NaiveForecaster", "params": {}, "reasoning": "default"})
```

Same `LLMForecaster` class. Zero code changes.

---

## 📊 Demo Results

### Real LLM (Groq/Llama-3.1) on Airline Dataset

| Prompt | LLM Selected | Fallback Used |
|--------|-------------|---------------|
| "simple baseline model" | ARIMA | ❌ |
| "seasonal model with strong trend" | NaiveForecaster | ✅ (AutoARIMA params failed validation) |
| "auto model, best parameters" | AutoARIMA | ❌ |

> **Note:** Prompt 2 triggered the fallback — AutoARIMA with LLM-suggested hyperparameters failed the validation gate on a small data slice. The system recovered gracefully. This is the validation gate working exactly as designed.

---

## 🧪 Design Principles

| Principle | How it's implemented |
|-----------|---------------------|
| **No hard dependencies** | `llm` is any callable — no framework required |
| **CI-safe** | Mock LLM needs zero API calls or network access |
| **Composable** | `LLMForecaster` is a `BaseForecaster` — works in any sktime pipeline |
| **Deterministic predictions** | LLM only involved at fit time, never at predict time |
| **Graceful degradation** | Validation gate + fallback means system never crashes |
| **Extensible** | Future: pipelines, tuning, ensembling — without redesign |

---

## 🗂️ Related Work

This prototype was built as part of broader ESoC 2026 contributions to the sktime agentic ecosystem:

- **sktime-mcp** — MCP server exposing sktime to LLM clients ([issues #112, #115, #117](https://github.com/sktime/sktime-mcp/issues))
- **sktime extension templates** — Forecasting metric templates ([PR #9838](https://github.com/sktime/sktime/issues/9838))
- **sktime issue #9721** — LLM-based forecasting agent discussion

---

## 👤 Author

**Nestroy Musoke** — CS student, Uganda Christian University  
ESoC 2026 applicant — sktime agentic track  
GitHub: [@NestroyMusoke](https://github.com/NestroyMusoke)

---

## 📄 License

BSD-3-Clause — same as sktime
