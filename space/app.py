"""Nyaya — retrieval demo for Hugging Face Spaces (CPU, no model).

Shows the sections of current Indian law a question resolves to, exactly the
context the reader model is given, plus the coverage gate's verdict. The
`nyaya` package and `data/canonical` are vendored next to this file by
scripts/37_publish_space.py; nothing is downloaded at start-up.
"""
from pathlib import Path

import gradio as gr

from nyaya.retrieval import COVERAGE_MIN_SCORE, load_statute_index

HERE = Path(__file__).resolve().parent
INDEX = load_statute_index(HERE / "data" / "canonical")

DISCLAIMER = (
    "⚖️ **Not legal advice.** Nyaya provides legal *information*. The practice of law in "
    "India is reserved to advocates enrolled under the Advocates Act, 1961. Consult a "
    "licensed advocate for anything consequential — free legal aid is available through "
    "NALSA / DLSA (helpline 15100)."
)
EXAMPLES = [
    ["Police FIR nahi likh rahi, kya karu?"],
    ["What is the punishment for cheque bounce?"],
    ["Are WhatsApp chats admissible as evidence in court?"],
    ["मेरा चेक बाउंस हो गया, मैं क्या कर सकता हूँ?"],
    ["Section 302 IPC ab BNS mein kya hai?"],
    ["makan malik security deposit wapas nahi kar raha"],
]


def ask(question: str, k: int):
    question = (question or "").strip()
    if not question:
        return "Ask a question to begin.", ""
    cov = INDEX.coverage(question)
    rows = INDEX.retrieve(question, k=int(k))
    if cov["covered"]:
        banner = (f"✅ **Inside the database.** Best statute match {cov['top_statute_score']:.1f}"
                  if cov["top_statute_score"] is not None else
                  "✅ **Explicit citation resolved.**")
    else:
        banner = (f"⚠️ **This question appears to fall outside the acts in this database** "
                  f"(best statute match {cov['top_statute_score']:.1f}, gate {COVERAGE_MIN_SCORE}). "
                  "The sections below are the nearest matches, not an answer. Rent, property, loans "
                  "and children's law are not indexed yet.")
    sections = "\n\n".join(
        (f"**{r['title']} — official guidance**  \n{(r.get('text') or '')[:700]}"
         if r["act_id"] == "procedures_kb" else
         f"**Section {r['section']} of the {r['act_name']}** — *{r.get('title') or ''}*  \n"
         f"{(r.get('text') or '')[:700]}")
        for r in rows) or "_No matching sections._"
    return banner, sections


with gr.Blocks(title="Nyaya — current Indian law, section by section") as demo:
    gr.Markdown(
        "# Nyaya\n### The statute retriever behind an open Indian legal guidance system\n"
        "Type a legal question in English, Hindi or Hinglish. You get the sections of **current** "
        "Indian law (BNS / BNSS / BSA post-July-2024, the Constitution and ten other acts) that the "
        "question resolves to — the same context the reader model is given. Exact citations, "
        "including old IPC / CrPC numbers, are bridged through the official mapping tables."
    )
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        q = gr.Textbox(label="Your question", lines=2, placeholder="Police FIR nahi likh rahi, kya karu?", scale=4)
        k = gr.Slider(1, 8, value=5, step=1, label="Sections", scale=1)
    go = gr.Button("Find the law", variant="primary")
    banner = gr.Markdown()
    out = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[q])
    go.click(ask, [q, k], [banner, out])
    q.submit(ask, [q, k], [banner, out])
    gr.Markdown(
        "---\nCoverage: 13 acts + the Constitution (2,528 sections), 1,257 official IPC↔BNS / "
        "CrPC↔BNSS / IEA↔BSA mappings, 70 procedural guidance notes. Absence is flagged, not hidden: "
        "about a quarter of real citizen questions (rent, property, loans, children) fall outside the "
        "indexed acts today. Code, data and every measurement, including the ones that were wrong: "
        "[github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model)."
    )

if __name__ == "__main__":
    demo.launch()
