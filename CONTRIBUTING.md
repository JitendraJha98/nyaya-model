# Contributing to Nyaya

Thank you for looking. Three kinds of contribution are especially welcome, in this order.

## 1. Add an act to the statute database

Most of the value here is current Indian law as clean, section-level data. To add a central act:

1. Find it on [India Code](https://www.indiacode.gov.in/) and note its exact title.
2. Add an entry under `api_acts:` in `configs/acts.yaml` (`act_id`, `act_name`, `indiacode_title`).
3. Run `python scripts/42_fetch_act_sections.py --act-id <act_id>`. It pulls every section
   through the India Code API, validates numbering, and writes `data/canonical/<act_id>.jsonl`.
4. Add the act's aliases to `src/nyaya/validators.py` (`ACT_ALIASES`) and its family to
   `src/nyaya/retrieval.py` (`_FAMILY_TO_ACT_ID`) so citations resolve.
5. Run `python scripts/35_coverage_probe.py` and `python -m pytest -q`, then open a pull request
   with the probe numbers before and after.

The gate is ≥ 98% clean sections. State law is out of scope for now: India Code's state
collections are inconsistent and the retriever assumes one act per name.

## 2. Report a wrong or missing answer

Open an issue with the question, the sections the retriever returned (`nyaya ask "<question>"`),
and the section you believe is right. Questions in Hindi and Hinglish are the most useful:
the evaluation set has few of them.

## 3. Improve a measured number

Every claim in this repository is a paired comparison with a bootstrap confidence interval on
committed predictions. A change to retrieval, prompting or the reader is accepted when:

- `python scripts/26_eval_v1_run.py` predictions for the new configuration are committed under
  `outputs/eval-v1/<label>/`, and
- `python scripts/27_compare_runs.py --a <baseline> --b <label>` is committed under `reports/`
  and its interval on fact recall excludes zero.

A tie is a valid, publishable result; it goes into `docs/RESULTS.md` like the others.

## House rules

- Statutory text is public domain; everything else you add must be Apache-2.0 compatible.
- No scraping of sites whose terms forbid it; India Code and legislative.gov.in are the sources.
- Nothing here is legal advice, and no wording should suggest otherwise.
- `pip install -e ".[dev]"` then `python -m pytest -q` before every pull request; CI runs the same.
