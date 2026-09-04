"""Nyaya demo — ask an Indian legal question, get a cited answer.

Runs the whole system end to end: retrieve statute sections, optionally rerank
them with a cross-encoder, and have the model answer from what it was given.

Deployment note (measured, not assumed)
---------------------------------------
The multilingual reranker costs ~1.6 s/pair on CPU, i.e. ~80 s per query at
depth 50 — unusable for a demo. It is therefore OFF unless a GPU is present.
Retrieval alone still works: BM25 + exact-citation lookup put the right
statute in the top 8 for ~81% of gold-bearing eval questions.

    python app.py                    # local, http://127.0.0.1:7860
    python app.py --no-model         # retrieval only, no weights downloaded
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nyaya.prompts import NYAYA_SYSTEM_PROMPT  # noqa: E402
from nyaya.retrieval import build_rag_prompt, format_context, load_statute_index  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

DISCLAIMER = (
    "⚖️ **Not legal advice.** Nyaya provides legal *information*. The practice "
    "of law in India is reserved to advocates enrolled under the Advocates Act, "
    "1961. Consult a licensed advocate for anything consequential — free legal "
    "aid is available through NALSA / DLSA."
)

EXAMPLES = [
    "Police FIR nahi likh rahi, kya karu?",
    "What is the punishment for murder under current Indian law?",
    "Are WhatsApp chats admissible as evidence in court?",
    "मेरा चेक बाउंस हो गया, मैं क्या कर सकता हूँ?",
    "Can my in-laws throw me out of the house during a matrimonial dispute?",
]


def _has_gpu() -> bool:
    try:
        import torch
    except (ImportError, OSError):  # a broken CUDA DLL raises OSError, not ImportError
        return False
    return torch.cuda.is_available()


def build(use_model: bool, use_rerank: bool):
    index = load_statute_index(str(ROOT / "data" / "canonical"))
    print(f"[app] statute index: {len(index.rows)} sections")

    if use_rerank:
        from nyaya.rerank import CrossEncoderReranker
        index.set_reranker(CrossEncoderReranker(depth=20))
        print("[app] reranking ON (GPU detected)")
    else:
        print("[app] reranking OFF — ~80 s/query on CPU, see module docstring")

    generate = None
    if use_model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = (torch.bfloat16
                 if _has_gpu() and torch.cuda.get_device_capability()[0] >= 8
                 else (torch.float16 if _has_gpu() else torch.float32))
        print(f"[app] loading {BASE_MODEL} as {dtype}")
        tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=dtype,
            device_map={"": 0} if _has_gpu() else None).eval()

        def generate(prompt: str) -> str:
            messages = [{"role": "system", "content": NYAYA_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=384, do_sample=False,
                                     pad_token_id=tok.eos_token_id or tok.pad_token_id)
            return tok.decode(out[0][enc["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

    return index, generate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-model", action="store_true",
                   help="retrieval only — shows the statutes, downloads no weights")
    p.add_argument("--rerank", action="store_true",
                   help="force reranking on (default: only when a GPU is present)")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    index, generate = build(use_model=not args.no_model,
                            use_rerank=args.rerank or _has_gpu())

    def answer(question: str, k: int):
        question = (question or "").strip()
        if not question:
            return "Ask a question to begin.", ""
        hits = index.retrieve(question, k=k)
        sources = "\n\n".join(
            f"**{h['act_name']} — Section {h['section']}**  \n"
            f"*{h.get('title') or ''}*  \n"
            f"{(h.get('text') or '')[:600]}"
            for h in hits) or "_No matching sections found._"
        if generate is None:
            return ("_Retrieval-only mode — the sections the model would be "
                    "given are shown on the right._"), sources
        return generate(build_rag_prompt(question, hits)), sources

    import gradio as gr

    with gr.Blocks(title="Nyaya — Indian legal guidance") as demo:
        gr.Markdown("# Nyaya\n### Open Indian legal guidance — BNS / BNSS / BSA, "
                    "post-July-2024. English, Hindi, Hinglish.")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="Your question", lines=2,
                               placeholder="Police FIR nahi likh rahi, kya karu?")
                k = gr.Slider(1, 12, value=args.k, step=1,
                              label="Statute sections to retrieve")
                go = gr.Button("Ask", variant="primary")
                out = gr.Markdown(label="Answer")
            with gr.Column(scale=2):
                gr.Markdown("#### Sections retrieved\n"
                            "The answer may only rely on these.")
                src = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=q)
        go.click(answer, [q, k], [out, src])
        q.submit(answer, [q, k], [out, src])
        gr.Markdown(
            "---\nAccuracy is honest about its limits: on the project's own "
            "benchmark the model answers **63%** of questions correctly when "
            "the right statute is retrieved, and **17%** when it is not — which "
            "is why the retrieved sections are shown alongside every answer."
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
