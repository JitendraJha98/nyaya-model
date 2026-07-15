"""Tests for statute retrieval (pure logic — no model, no network)."""

from pathlib import Path

import pytest

from nyaya.retrieval import (StatuteIndex, build_rag_prompt,
                             build_rag_training_record, format_context,
                             load_statute_index)

CANONICAL_DIR = Path(__file__).resolve().parents[1] / "data" / "canonical"

ROWS = [
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "318",
     "title": "Cheating", "text": "Whoever, by deceiving any person, fraudulently or dishonestly "
     "induces the person so deceived to deliver any property commits cheating. Punishment "
     "extends to seven years imprisonment and fine.", "chapter": "CHAPTER XVII"},
    {"act_id": "bns_2023", "act_name": "Bharatiya Nyaya Sanhita, 2023", "section": "303",
     "title": "Theft", "text": "Whoever intends to take dishonestly any movable property out of "
     "the possession of any person without consent commits theft.", "chapter": "CHAPTER XVII"},
    {"act_id": "bnss_2023", "act_name": "Bharatiya Nagarik Suraksha Sanhita, 2023", "section": "173",
     "title": "Information in cognizable cases", "text": "Every information relating to the "
     "commission of a cognizable offence, irrespective of the area where the offence is "
     "committed, may be given to an officer in charge of a police station. First Information "
     "Report registration.", "chapter": "CHAPTER XIII"},
    {"act_id": "rti_2005", "act_name": "Right to Information Act, 2005", "section": "7",
     "title": "Disposal of request", "text": "Request for information shall be disposed of "
     "within thirty days of the receipt of the request by the public information officer.",
     "chapter": "CHAPTER II"},
]

MAPPINGS = [
    {"old_act": "IPC", "old_section": "420", "new_act": "BNS", "new_section": "318", "note": None},
    {"old_act": "CrPC", "old_section": "154", "new_act": "BNSS", "new_section": "173", "note": None},
]


@pytest.fixture(scope="module")
def index():
    return StatuteIndex(ROWS, MAPPINGS)


class TestExactCitationLookup:
    def test_query_naming_a_section_retrieves_it_first(self, index):
        hits = index.retrieve("What is the punishment under Section 318 of the BNS?", k=2)
        assert hits[0]["section"] == "318" and hits[0]["act_id"] == "bns_2023"

    def test_old_law_query_maps_to_new_section(self, index):
        hits = index.retrieve("I was cheated — is IPC Section 420 still a thing?", k=2)
        assert hits[0]["section"] == "318" and hits[0]["act_id"] == "bns_2023"

    def test_devanagari_citation(self, index):
        hits = index.retrieve("धारा 173 BNSS के तहत FIR कैसे होगी?", k=2)
        assert hits[0]["section"] == "173" and hits[0]["act_id"] == "bnss_2023"

    def test_bare_article_implies_constitution(self, index):
        # rti section 21 makes the bare number 21 ambiguous — the "Article"
        # marker itself must bind the Constitution
        rows = ROWS + [
            {"act_id": "constitution_1950", "act_name": "Constitution of India",
             "section": "21", "title": "Protection of life and personal liberty",
             "text": "No person shall be deprived of his life or personal liberty "
             "except according to procedure established by law.", "chapter": "PART III"},
            {"act_id": "rti_2005", "act_name": "Right to Information Act, 2005",
             "section": "21", "title": "Protection of action taken in good faith",
             "text": "No suit or prosecution shall lie against any person for anything "
             "done in good faith under this Act.", "chapter": "CHAPTER VI"},
        ]
        idx = StatuteIndex(rows, MAPPINGS)
        hits = idx.retrieve("Article 21", k=2)
        assert hits[0]["act_id"] == "constitution_1950" and hits[0]["section"] == "21"
        hits_hi = idx.retrieve("अनुच्छेद 21 क्या कहता है?", k=2)
        assert hits_hi[0]["act_id"] == "constitution_1950"


class TestLexicalRetrieval:
    def test_concept_query_finds_relevant_section(self, index):
        hits = index.retrieve("Someone deceived my father into delivering property — what offence?", k=2)
        assert hits[0]["section"] == "318"

    def test_procedural_query(self, index):
        hits = index.retrieve("How many days does the public information officer have to reply?", k=2)
        assert hits[0]["act_id"] == "rti_2005"

    def test_k_limits_results(self, index):
        assert len(index.retrieve("property", k=2)) <= 2

    def test_no_signal_returns_something_not_crash(self, index):
        assert isinstance(index.retrieve("zzz qqq xyzzy", k=3), list)

    def test_title_outweighs_body_frequency(self, index):
        # 303's body says "theft" once in the title-position; a decoy body
        # that repeats the word must not outrank the section titled for it
        rows = ROWS + [{"act_id": "bnss_2023", "act_name": "Bharatiya Nagarik Suraksha "
                        "Sanhita, 2023", "section": "999", "title": "Procedure on complaint",
                        "text": "theft theft theft theft theft reported to the magistrate "
                        "in cases of theft the procedure for theft complaints applies.",
                        "chapter": "X"}]
        idx = StatuteIndex(rows, MAPPINGS)
        hits = idx.retrieve("what counts as theft?", k=1)
        assert hits[0]["section"] == "303"


class TestQueryExpansion:
    def test_hindi_legal_term_reaches_english_statute(self, index):
        rows = ROWS + [{"act_id": "bnss_2023", "act_name": "Bharatiya Nagarik Suraksha "
                        "Sanhita, 2023", "section": "480", "title": "In what cases bail to be taken",
                        "text": "When any person accused of a bailable offence is arrested or "
                        "detained without warrant he shall be released on bail.", "chapter": "XXXV"}]
        idx = StatuteIndex(rows, MAPPINGS)
        hits = idx.retrieve("मुझे जमानत कैसे मिलेगी?", k=2)
        assert any(h["section"] == "480" for h in hits)

    def test_hinglish_lay_term(self, index):
        hits = index.retrieve("Mere saath dhokha hua, paise le kar bhaag gaya", k=2)
        assert any(h["section"] == "318" for h in hits)

    def test_cheque_bounce_lay_phrase(self, index):
        rows = ROWS + [{"act_id": "ni_act_1881", "act_name": "Negotiable Instruments Act, 1881",
                        "section": "138", "title": "Dishonour of cheque for insufficiency, "
                        "etc., of funds in the account",
                        "text": "Where any cheque drawn by a person is returned by the bank "
                        "unpaid because of insufficiency of funds such person shall be deemed "
                        "to have committed an offence.", "chapter": "XVII"}]
        idx = StatuteIndex(rows, MAPPINGS)
        hits = idx.retrieve("My cheque bounced, what can I do?", k=2)
        assert any(h["section"] == "138" for h in hits)


@pytest.mark.skipif(not any(CANONICAL_DIR.glob("bns_*.jsonl")),
                    reason="canonical statute DB not built")
class TestRealStatuteDB:
    @pytest.fixture(scope="class")
    def real_index(self):
        return load_statute_index(CANONICAL_DIR)

    def test_indexes_every_canonical_row(self, real_index):
        assert len(real_index.rows) > 2000  # 14 acts + Constitution, sans mappings

    def test_murder_section_lookup(self, real_index):
        hits = real_index.retrieve("What is the punishment under Section 103 of the BNS?", k=4)
        assert hits[0]["act_id"] == "bns_2023" and hits[0]["section"] == "103"

    def test_ipc_302_maps_to_bns_103(self, real_index):
        hits = real_index.retrieve("Is IPC Section 302 for murder still valid?", k=4)
        assert any(h["act_id"] == "bns_2023" and h["section"] == "103" for h in hits)

    def test_article_21_lookup(self, real_index):
        hits = real_index.retrieve("What does Article 21 of the Constitution guarantee?", k=4)
        assert any(h["act_id"] == "constitution_1950" and h["section"] == "21" for h in hits)


def _training_record(rec_id="gen_000001_ab_01", sections=("bns_2023:303",),
                     question="Meri cycle chori ho gayi, kya karun?"):
    return {
        "id": rec_id,
        "messages": [
            {"role": "system", "content": "You are a legal assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Yeh theft hai — Section 303 of the "
             "Bharatiya Nyaya Sanhita, 2023 ke tahat report karein."},
        ],
        "metadata": {"task_type": "grounded_qa", "language": "hinglish",
                     "source_sections": list(sections)},
    }


class TestBuildRagTrainingRecord:
    def test_gold_section_always_in_context(self, index):
        # BM25 for this vague query may miss 303 — injection must guarantee it
        out = build_rag_training_record(_training_record(), index, k=2)
        assert "Whoever intends to take dishonestly any movable property" in \
            out["messages"][1]["content"]

    def test_user_message_is_rag_prompt_answer_untouched(self, index):
        rec = _training_record()
        out = build_rag_training_record(rec, index, k=2)
        assert out["messages"][1]["content"].rstrip().endswith(rec["messages"][1]["content"])
        assert out["messages"][2] == rec["messages"][2]
        assert rec["messages"][1]["content"] == "Meri cycle chori ho gayi, kya karun?"

    def test_deterministic(self, index):
        a = build_rag_training_record(_training_record(), index, k=3)
        b = build_rag_training_record(_training_record(), index, k=3)
        assert a == b

    def test_gold_position_varies_across_ids(self, index):
        # anti positional-bias: gold must not always sit first in the context
        positions = set()
        for i in range(12):
            rec = _training_record(rec_id=f"gen_{i:06d}_xx_01",
                                   question="Section 7 RTI aur meri cycle chori ho gayi")
            out = build_rag_training_record(rec, index, k=3)
            ctx = out["messages"][1]["content"]
            positions.add(ctx.index("movable property"))
        assert len(positions) > 1

    def test_no_source_sections_uses_plain_retrieval(self, index):
        rec = _training_record(sections=())
        rec["metadata"]["task_type"] = "safety_abstention"
        out = build_rag_training_record(rec, index, k=2)
        assert "Relevant provisions" in out["messages"][1]["content"]

    def test_metadata_records_context(self, index):
        out = build_rag_training_record(_training_record(), index, k=2)
        assert "bns_2023:303" in out["metadata"]["rag"]["context_keys"]
        assert out["metadata"]["rag"]["k"] == 2


class TestRagPrompt:
    def test_prompt_embeds_context_and_question(self, index):
        hits = index.retrieve("Section 318 BNS", k=2)
        prompt = build_rag_prompt("Kya IPC 420 abhi bhi lagta hai?", hits)
        assert "deceiving any person" in prompt
        assert prompt.rstrip().endswith("Kya IPC 420 abhi bhi lagta hai?")
        assert "do not cite any section not shown above" in prompt


class TestContextPacking:
    def _long_row(self, section, words):
        return {"act_id": "mv_act_1988", "act_name": "Motor Vehicles Act, 1988",
                "section": section, "title": f"Long provision {section}",
                "text": " ".join(f"w{i}" for i in range(words)), "chapter": "X"}

    def test_budget_drops_overflow_rows(self, index):
        from nyaya.retrieval import pack_rows
        rows = [self._long_row("1", 1500), self._long_row("2", 1500),
                self._long_row("3", 200)]
        packed = pack_rows(rows, budget_words=2000)
        sections = [r["section"] for r in packed]
        assert sections == ["1", "3"]  # row 2 busts the budget, row 3 still fits

    def test_single_huge_row_truncated_not_dropped(self, index):
        from nyaya.retrieval import pack_rows
        packed = pack_rows([self._long_row("1", 5000)], budget_words=2000)
        assert len(packed) == 1
        assert len(packed[0]["text"].split()) <= 2100

    def test_must_keep_rows_truncate_instead_of_drop(self, index):
        from nyaya.retrieval import pack_rows
        rows = [self._long_row("1", 1500), self._long_row("2", 1500)]
        packed = pack_rows(rows, budget_words=2000,
                           must_keys={"mv_act_1988:2"})
        assert [r["section"] for r in packed] == ["1", "2"]

    def test_training_record_stays_within_budget_with_gold(self, index):
        rec = _training_record()
        out = build_rag_training_record(rec, index, k=8, budget_words=200)
        user = next(m["content"] for m in out["messages"] if m["role"] == "user")
        assert len(user.split()) < 400  # budget + instructions + question
        assert "movable property" in user  # gold survived packing


class TestFormatContext:
    def test_contains_verbatim_text_and_citation_form(self, index):
        hits = index.retrieve("Section 318 BNS", k=1)
        ctx = format_context(hits)
        assert "Section 318" in ctx
        assert "Bharatiya Nyaya Sanhita" in ctx
        assert "deceiving any person" in ctx
