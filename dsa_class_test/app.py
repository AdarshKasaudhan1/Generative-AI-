"""Custom-Style GPT-2 Fine-Tuning Lab - Streamlit Web Dashboard.

Provides an interactive GUI for dataset inspection, model fine-tuning with live metrics,
side-by-side base vs. fine-tuned text generation comparison, and loss auditing.
"""

import os
import glob
import json
import time
import html
import streamlit as st
import pandas as pd
import torch

from src.dataset import (
    load_dataset_from_file,
    split_text_into_samples,
    prepare_datasets,
    clean_and_normalize_text,
    extract_text_from_pdf,
)
from src.model import load_gpt2_model_and_tokenizer
from src.trainer import train_gpt2
from src.generator import generate_text, compare_models


st.set_page_config(
    page_title="Custom-Style GPT-2 Fine-Tuning Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #60A5FA;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #334155;
        text-align: center;
        color: #F8FAFC;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-base {
        background-color: #334155;
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #475569;
        display: inline-block;
        margin-bottom: 6px;
    }
    .badge-ft {
        background-color: #1E3A8A;
        color: #BFDBFE;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.82rem;
        border: 1px solid #3B82F6;
        display: inline-block;
        margin-bottom: 6px;
    }
    .output-box {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border-radius: 8px;
        border: 1px solid #334155;
        border-left: 5px solid #3B82F6 !important;
        padding: 16px;
        font-family: Consolas, 'Courier New', Monaco, monospace;
        font-size: 0.94rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-break: break-word;
        min-height: 200px;
        max-height: 420px;
        overflow-y: auto;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Base GPT-2 (124M)...")
def get_cached_base_model():
    """Cache the base GPT-2 model in memory across runs."""
    model, tokenizer, meta = load_gpt2_model_and_tokenizer("gpt2")
    return model, tokenizer, meta


@st.cache_resource(show_spinner="Loading Fine-Tuned Checkpoint...")
def get_cached_finetuned_model(checkpoint_path: str):
    """Cache fine-tuned checkpoints by directory path."""
    model, tokenizer, meta = load_gpt2_model_and_tokenizer(checkpoint_path)
    return model, tokenizer, meta


def list_saved_checkpoints() -> list:
    """Return all directories under checkpoints/ containing model weights."""
    checkpoints = []
    if os.path.exists("checkpoints"):
        for entry in os.listdir("checkpoints"):
            full_path = os.path.join("checkpoints", entry)
            if os.path.isdir(full_path):
                weights_exist = os.path.exists(os.path.join(full_path, "model.safetensors")) or \
                                os.path.exists(os.path.join(full_path, "pytorch_model.bin"))
                if weights_exist:
                    checkpoints.append(entry)
    return sorted(checkpoints)


# Header Section
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">🧬 Custom-Style GPT-2 Fine-Tuning Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">End-to-End Domain Adaptation: Dataset Curation &rarr; Causal LM Training &rarr; Before-and-After Generation Audit</div>', unsafe_allow_html=True)
with col_h2:
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU (Standard)"
    st.info(f"**Runtime Hardware**: {device_name}")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 1. Dataset Preprocessing",
    "⚙️ 2. Fine-Tuning Lab",
    "⚖️ 3. Side-by-Side Generation",
    "📈 4. Loss Monitoring & Checkpoints",
    "🔬 5. Architecture & Math"
])

# ==============================================================================
# TAB 1: DATASET PREPROCESSING & TOKENIZATION
# ==============================================================================
with tab1:
    st.subheader("Dataset Ingestion & Tokenizer Inspection")
    st.write("Upload a custom text file or select one of the built-in domain datasets to inspect BPE tokens and causal splits.")

    col_ds1, col_ds2 = st.columns([1, 2])
    with col_ds1:
        data_source = st.radio(
            "Select Data Source:",
            ["Preset: Campus Announcements", "Preset: Cyberpunk Catalog", "Preset: Micro Stories", "Upload File (TXT, CSV, PDF)"]
        )

        raw_dataset_text = ""
        if data_source == "Preset: Campus Announcements":
            if os.path.exists("data/campus_announcements.txt"):
                raw_dataset_text = load_dataset_from_file("data/campus_announcements.txt")
        elif data_source == "Preset: Cyberpunk Catalog":
            if os.path.exists("data/cyberpunk_products.txt"):
                raw_dataset_text = load_dataset_from_file("data/cyberpunk_products.txt")
        elif data_source == "Preset: Micro Stories":
            if os.path.exists("data/micro_stories.txt"):
                raw_dataset_text = load_dataset_from_file("data/micro_stories.txt")
        else:
            uploaded_file = st.file_uploader("Upload dataset file:", type=["txt", "csv", "pdf"])
            if uploaded_file is not None:
                filename = uploaded_file.name.lower()
                if filename.endswith(".pdf"):
                    raw_dataset_text = extract_text_from_pdf(uploaded_file)
                    if raw_dataset_text.startswith("[Error"):
                        st.error(raw_dataset_text)
                        raw_dataset_text = ""
                    else:
                        st.success(f"Extracted text from PDF ({len(raw_dataset_text)} characters).")
                elif filename.endswith(".csv"):
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        text_cols = [c for c in df_upload.columns if df_upload[c].dtype == "object"]
                        if text_cols:
                            raw_dataset_text = "\n\n".join(df_upload[text_cols[0]].dropna().astype(str).tolist())
                            st.success(f"Loaded column '{text_cols[0]}' with {len(df_upload)} records.")
                        else:
                            st.error("No text columns found in CSV.")
                    except Exception as e:
                        st.error(f"Error parsing CSV: {e}")
                else:
                    raw_dataset_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

        max_seq_len = st.slider("Max Block Token Length ($L$):", min_value=32, max_value=256, value=128, step=16)

    with col_ds2:
        if raw_dataset_text.strip():
            base_model, tokenizer, _ = get_cached_base_model()
            samples = split_text_into_samples(raw_dataset_text)
            token_lengths = [len(tokenizer.encode(s)) for s in samples] if samples else [0]
            avg_tok = round(sum(token_lengths) / max(1, len(token_lengths)), 1)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", len(samples))
            m2.metric("Total Tokens", f"{sum(token_lengths):,}")
            m3.metric("Avg Tokens/Sample", avg_tok)
            m4.metric("Max Sample Length", max(token_lengths))

            st.markdown("#### Sample Explorer")
            sample_idx = st.selectbox("Select Sample to inspect:", range(len(samples)), format_func=lambda i: f"Sample #{i + 1} ({token_lengths[i]} tokens)")
            selected_text = samples[sample_idx]

            line_count = len(selected_text.splitlines())
            box_height = min(220, max(100, line_count * 22))
            st.text_area("Cleaned Sample Text:", selected_text, height=box_height)

            tokens = tokenizer.tokenize(selected_text)
            token_ids = tokenizer.encode(selected_text)

            with st.expander(f"🔍 Byte-Pair Encoding (BPE) Token Breakdown ({len(tokens)} tokens)"):
                df_tokens = pd.DataFrame({
                    "Index": range(len(tokens[:60])),
                    "Token String": [repr(t) for t in tokens[:60]],
                    "Token ID": token_ids[:60]
                })
                st.dataframe(df_tokens.T, use_container_width=True)
        else:
            st.info("Upload a file above, or select one of the 3 Curated Samples to try it immediately.")


# ==============================================================================
# TAB 2: FINE-TUNING LAB & HYPERPARAMETER CONTROLS
# ==============================================================================
with tab2:
    st.subheader("Training Configuration & Live Execution")
    st.write("Tune GPT-2 parameters on your chosen corpus using full backpropagation, layer freezing, or LoRA.")

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        dataset_options = ["Campus Announcements", "Cyberpunk Catalog", "Micro Stories"]
        if "custom_dataset" in st.session_state:
            dataset_options.append("Current Uploaded Custom Dataset")

        training_dataset_name = st.selectbox("Training Dataset:", dataset_options)
        epochs = st.slider("Epochs:", min_value=1, max_value=10, value=5)
        batch_size = st.select_slider("Batch Size:", options=[1, 2, 4], value=2)
        lr = st.select_slider("Learning Rate (AdamW):", options=[1e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4], value=8e-5, format_func=lambda x: f"{x:.0e}")
        freeze_blocks = st.slider("Freeze Early Transformer Blocks (0 = none):", 0, 10, 0)

        clean_name = training_dataset_name.lower().replace(" ", "_")
        default_ck_name = f"custom_{clean_name}_run"
        checkpoint_name = st.text_input("Checkpoint Output Subdirectory:", value=default_ck_name)

        start_train_btn = st.button("🚀 Start Fine-Tuning Run", type="primary", use_container_width=True)

    with col_t2:
        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = st.progress(0.0)

        if start_train_btn:
            # Resolve data content
            if "Campus" in training_dataset_name:
                raw_text = load_dataset_from_file("data/campus_announcements.txt")
            elif "Cyberpunk" in training_dataset_name:
                raw_text = load_dataset_from_file("data/cyberpunk_products.txt")
            elif "Micro" in training_dataset_name:
                raw_text = load_dataset_from_file("data/micro_stories.txt")
            else:
                raw_text = st.session_state.get("custom_dataset", "")

            if not raw_text.strip():
                st.error("Training text is empty. Please select or upload a dataset first.")
            else:
                status_placeholder.info("Initializing Base GPT-2 and preparing causal datasets...")
                model, tokenizer, meta = load_gpt2_model_and_tokenizer("gpt2", freeze_early_layers=freeze_blocks)
                train_ds, val_ds, stats = prepare_datasets(raw_text, tokenizer, max_length=128, val_split=0.15)

                step_loss_data = []

                def ui_callback(event):
                    if event["event"] == "step":
                        progress_val = min(1.0, event["step"] / max(1, event["total_steps"]))
                        progress_bar.progress(progress_val)
                        step_loss_data.append({"Step": event["step"], "Loss": event["loss"]})
                        status_placeholder.markdown(
                            f"**Epoch {event['epoch']}/{event['epochs']}** | Step {event['step']}/{event['total_steps']} | Batch Loss: `{event['loss']:.4f}` | LR: `{event['lr']:.2e}`"
                        )
                        if len(step_loss_data) > 1:
                            chart_placeholder.line_chart(pd.DataFrame(step_loss_data).set_index("Step"), height=240)
                    elif event["event"] == "epoch":
                        val_info = f" | Val Loss: `{event['val_loss']:.4f}` | PPL: `{event['perplexity']:.2f}`" if event["val_loss"] else ""
                        status_placeholder.markdown(
                            f"✅ Completed Epoch {event['epoch']}/{event['epochs']} (Train Loss: `{event['train_loss']:.4f}`{val_info})"
                        )

                output_dir = os.path.join("checkpoints", checkpoint_name.strip())
                status_placeholder.info(f"Training in progress on {meta['device']}...")

                history = train_gpt2(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    output_dir=output_dir,
                    progress_callback=ui_callback
                )

                progress_bar.progress(1.0)
                status_placeholder.success(
                    f"🎉 Fine-tuning finished in {history['total_training_time_sec']}s! Saved to `{output_dir}`.\nFinal Train Loss: `{history['final_train_loss']}` | Final Perplexity: `{history['final_perplexity']}`"
                )
                st.cache_resource.clear()
        else:
            status_placeholder.info("Configure training hyperparameters on the left and click 'Start Fine-Tuning Run'.")


# ==============================================================================
# TAB 3: SIDE-BY-SIDE GENERATION COMPARISON
# ==============================================================================
with tab3:
    st.subheader("Before-vs-After Generation Audit")
    st.write("Compare completions between Base Pretrained GPT-2 and your Fine-Tuned checkpoint under identical sampling settings.")

    checkpoints_available = list_saved_checkpoints()
    if not checkpoints_available:
        st.warning("No fine-tuned checkpoints found in `checkpoints/`. Run training in Tab 2 or use pre-existing checkpoints.")
        st.stop()

    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        selected_checkpoint = st.selectbox(
            "Select Fine-Tuned Checkpoint:",
            checkpoints_available,
            index=0
        )
        checkpoint_full_path = os.path.join("checkpoints", selected_checkpoint)

        preset_cols = st.columns(3)
        prompt_preset = None
        if preset_cols[0].button("Campus", use_container_width=True):
            prompt_preset = "[CAMPUS NOTICE]\nDATE: November 15, 2026\nDEPARTMENT: Office of Student Affairs\nSUBJECT:"
        if preset_cols[1].button("Cyberpunk", use_container_width=True):
            prompt_preset = "[PRODUCT: Quantum Neural Interface]\nMANUFACTURER: Arasaka Sub-orbital Dynamics\nSPECS:"
        if preset_cols[2].button("Story", use_container_width=True):
            prompt_preset = "[STORY: The Mechanical Forest]\nDeep beneath the abandoned copper pines"

        default_prompt = prompt_preset or "[CAMPUS NOTICE]\nDATE: November 15, 2026\nDEPARTMENT: Department of Artificial Intelligence\nSUBJECT:"
        user_prompt = st.text_area("Prompt Input:", value=default_prompt, height=130)

        with st.expander("🛠️ Generation Hyperparameters", expanded=False):
            max_new_tokens = st.slider("Max New Tokens:", 20, 200, 80, 10)
            min_new_tokens = st.slider("Min New Tokens:", 0, 50, 10, 5)
            temperature = st.slider("Temperature ($T$):", 0.1, 1.5, 0.75, 0.05)
            top_p = st.slider("Top-p (Nucleus):", 0.1, 1.0, 0.95, 0.05)
            top_k = st.slider("Top-k:", 1, 100, 50, 5)
            rep_penalty = st.slider("Repetition Penalty:", 1.0, 2.0, 1.15, 0.05)
            stop_at_eos = st.checkbox("Stop at `<|endoftext|>` (EOS)", value=True)
            seed = st.number_input("Random Seed:", value=42, step=1)

        generate_btn = st.button("⚡ Generate Comparison", type="primary", use_container_width=True)

    with col_p2:
        if generate_btn or "last_comparison" in st.session_state:
            if generate_btn:
                with st.spinner("Executing dual inference on Base and Fine-Tuned models..."):
                    base_model, base_tok, _ = get_cached_base_model()
                    ft_model, ft_tok, _ = get_cached_finetuned_model(checkpoint_full_path)

                    res = compare_models(
                        base_model=base_model,
                        base_tokenizer=base_tok,
                        ft_model=ft_model,
                        ft_tokenizer=ft_tok,
                        prompt=user_prompt,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=rep_penalty,
                        stop_at_eos=stop_at_eos,
                        seed=int(seed)
                    )
                    st.session_state["last_comparison"] = res
            else:
                res = st.session_state["last_comparison"]

            col_out1, col_out2 = st.columns(2)
            with col_out1:
                st.markdown('<span class="badge-base">BASE PRETRAINED GPT-2</span>', unsafe_allow_html=True)
                base_text_html = html.escape(res["base"]["full_text"])
                st.markdown(f'<div class="output-box">{base_text_html}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ Latency: **{res['base']['latency_sec']}s** | Speed: **{res['base']['tokens_per_sec']} tok/s** | Tokens: **{res['base']['tokens_generated']}**")
                with st.expander("📋 Copy Raw Text"):
                    st.text_area("Base Output Raw", value=res["base"]["full_text"], height=130, key="raw_base_out", label_visibility="collapsed")

            with col_out2:
                st.markdown(f'<span class="badge-ft">FINE-TUNED ({selected_checkpoint})</span>', unsafe_allow_html=True)
                ft_text_html = html.escape(res["finetuned"]["full_text"])
                st.markdown(f'<div class="output-box">{ft_text_html}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ Latency: **{res['finetuned']['latency_sec']}s** | Speed: **{res['finetuned']['tokens_per_sec']} tok/s** | Tokens: **{res['finetuned']['tokens_generated']}**")
                with st.expander("📋 Copy Raw Text"):
                    st.text_area("Fine-Tuned Output Raw", value=res["finetuned"]["full_text"], height=130, key="raw_ft_out", label_visibility="collapsed")

            # Style Shift Analysis Card
            shift = res["style_shift"]
            st.markdown("---")
            st.markdown("#### 🔍 Quantitative Style Shift Analysis")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Base Word Count", shift["base_word_count"])
            s2.metric("Fine-Tuned Word Count", shift["finetuned_word_count"])
            s3.metric("Shared Vocabulary", shift["shared_vocabulary_count"])
            s4.metric("Structured Format", "Adopted ✅" if shift["adopted_structured_format"] else "Generic Prose ❌")

            if shift["finetuned_unique_words"]:
                st.markdown("**Domain-Specific Lexicon Emerged in Fine-Tuned Output:**")
                st.write(", ".join([f"`{w}`" for w in shift["finetuned_unique_words"][:12]]))
        else:
            st.info("Select a checkpoint and prompt on the left, then click 'Generate Comparison'.")


# ==============================================================================
# TAB 4: LOSS MONITORING & SAVED CHECKPOINTS
# ==============================================================================
with tab4:
    st.subheader("Training Convergence & History Audit")
    checkpoints_with_history = []
    for ck in list_saved_checkpoints():
        h_path = os.path.join("checkpoints", ck, "training_history.json")
        if os.path.exists(h_path):
            checkpoints_with_history.append(ck)

    if checkpoints_with_history:
        ck_to_inspect = st.selectbox("Select Checkpoint History:", checkpoints_with_history)
        h_file = os.path.join("checkpoints", ck_to_inspect, "training_history.json")
        with open(h_file, "r", encoding="utf-8") as f:
            h_data = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Train Loss", h_data.get("final_train_loss", "N/A"))
        c2.metric("Final Perplexity", h_data.get("final_perplexity", "N/A"))
        c3.metric("Training Duration", f"{h_data.get('total_training_time_sec', 0)}s")
        c4.metric("Configured Epochs", h_data.get("config", {}).get("epochs", "N/A"))

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### Step-by-Step Training Loss")
            if "step_losses" in h_data and h_data["step_losses"]:
                df_step = pd.DataFrame({
                    "Optimization Step": range(1, len(h_data["step_losses"]) + 1),
                    "Cross-Entropy Loss": h_data["step_losses"]
                })
                st.line_chart(df_step.set_index("Optimization Step"), height=250)

        with col_g2:
            st.markdown("##### Epoch Loss & Perplexity Trajectory")
            if "epoch_losses" in h_data and h_data["epoch_losses"]:
                df_epoch = pd.DataFrame({
                    "Epoch": range(1, len(h_data["epoch_losses"]) + 1),
                    "Train Loss": h_data["epoch_losses"],
                    "Validation Loss": h_data.get("val_losses", [None] * len(h_data["epoch_losses"]))
                })
                st.line_chart(df_epoch.set_index("Epoch"), height=250)

        with st.expander("📄 Full JSON History Record"):
            st.json(h_data)
    else:
        st.info("No training histories found yet.")


# ==============================================================================
# TAB 5: ARCHITECTURE & MATH
# ==============================================================================
with tab5:
    st.subheader("Architectural Theory & Methodology")
    st.markdown(r"""
### 1. Causal Language Modeling Formulation
Autoregressive language models predict the conditional probability of token $x_t$ given context $x_{<t}$:
$$P_\theta(X) = \prod_{t=1}^T P_\theta(x_t \mid x_1, x_2, \dots, x_{t-1})$$

The fine-tuning optimization minimizes cross-entropy loss over custom text corpus $\mathcal{D}_{\text{custom}}$:
$$\mathcal{L}_{\text{CLM}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T_i} \log P_\theta(x_{i,t} \mid x_{i,<t})$$

### 2. Domain Adaptation vs. Catastrophic Forgetting
- **Base GPT-2**: Pretrained on WebText ($\approx 40\text{ GB}$ of internet links). Output tendencies are general, conversational, and generic expository prose.
- **Domain Fine-Tuning**: Shifts model activations toward domain syntax, metadata tags (e.g., `[CAMPUS NOTICE]`, `DATE:`, `DEPARTMENT:`), specialized vocabulary, and formatted field delimiters.
- **Perplexity Metric**:
$$\text{PPL} = \exp\left( \mathcal{L}_{\text{CLM}} \right)$$
A lower perplexity indicates higher confidence and fluency on the custom domain distribution.

### 3. Byte-Pair Encoding (BPE) Tokenization
GPT-2 employs a byte-level BPE tokenizer with a vocabulary size of $V = 50,257$. In our pipeline:
1. Continuous corpora are split into discrete semantic units using structural pattern matching.
2. Special token `<|endoftext|>` is appended to separate documents.
3. Targets are masked at padding locations (`label = -100`) so gradient updates only penalize generated tokens.
""")
