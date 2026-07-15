"""Tests for statute retrieval (pure logic — no model, no network)."""

from pathlib import Path

import pytest

from nyaya.retrieval import (StatuteIndex, build_rag_prompt, format_context,
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


class TestRagPrompt:
    def test_prompt_embeds_context_and_question(self, index):
        hits = index.retrieve("Section 318 BNS", k=2)
        prompt = build_rag_prompt("Kya IPC 420 abhi bhi lagta hai?", hits)
        assert "deceiving any person" in prompt
        assert prompt.rstrip().endswith("Kya IPC 420 abhi bhi lagta hai?")
        assert "do not cite any section not shown above" in prompt


class TestFormatContext:
    def test_contains_verbatim_text_and_citation_form(self, index):
        hits = index.retrieve("Section 318 BNS", k=1)
        ctx = format_context(hits)
        assert "Section 318" in ctx
        assert "Bharatiya Nyaya Sanhita" in ctx
        assert "deceiving any person" in ctx
