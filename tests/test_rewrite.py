from nyaya.rewrite import REWRITE_PROMPT, needs_rewrite, rewrite_query


def test_devanagari_and_hinglish_need_rewrite_english_does_not():
    assert needs_rewrite("मकान मालिक डिपॉजिट वापस नहीं कर रहा")
    assert needs_rewrite("makan malik deposit wapas nahi kar raha")
    assert not needs_rewrite("landlord is not returning my security deposit")
    assert not needs_rewrite("What is the punishment for cheque bounce?")


def test_prompt_asks_for_one_line_of_statutory_english_and_no_answer():
    lowered = REWRITE_PROMPT.lower()
    assert "one line" in lowered
    assert "do not answer" in lowered
    assert "{question}" in REWRITE_PROMPT


def test_rewrite_appends_the_generated_line_to_the_original():
    def fake_generate(prompt):
        assert "makan malik" in prompt
        return "Rewritten: landlord not returning security deposit; recovery of money\nignored second line"

    out = rewrite_query("makan malik deposit wapas nahi kar raha", fake_generate)
    assert out.startswith("makan malik deposit wapas nahi kar raha ")
    assert "landlord not returning security deposit" in out
    assert "ignored second line" not in out


def test_english_questions_never_call_the_generator():
    def explode(prompt):
        raise AssertionError("generator must not be called for English")

    q = "What is the punishment for cheque bounce?"
    assert rewrite_query(q, explode) == q


def test_empty_or_runaway_generation_falls_back_to_the_original():
    q = "police FIR nahi likh rahi kya karu"
    assert rewrite_query(q, lambda p: "") == q
    assert rewrite_query(q, lambda p: "x" * 500) == q
