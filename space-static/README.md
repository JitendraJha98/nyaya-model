---
title: Nyaya
emoji: ⚖️
colorFrom: gray
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: Current Indian law, section by section
---

# Nyaya — retrieval demo (static, runs in your browser)

The statute retriever behind the Nyaya Indian legal guidance system, ported to
JavaScript and run entirely client-side over an exported copy of the statute
database. Type a question in English, Hindi or Hinglish and see the sections of
current law (BNS / BNSS / BSA post-July-2024, the Constitution and twenty-four other acts)
it resolves to, with a coverage verdict. `scripts/39_build_static_demo.py --check`
in the repository verifies that this port returns the same sections as the Python
retriever on real questions.

This page is retrieval only. The full system reads these sections with
`Qwen/Qwen3-4B-Instruct-2507` (52.0% fact recall on the project's benchmark); the
project's own fine-tune, [nyaya-3b-v3](https://huggingface.co/NyayaLabs98/nyaya-3b-v3),
is statistically tied with its base and is not the default.
Code and data: [github.com/JitendraJha98/nyaya-model](https://github.com/JitendraJha98/nyaya-model).
**Not legal advice.**
