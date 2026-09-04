// Nyaya statute retriever — a line-for-line JavaScript port of src/nyaya/retrieval.py
// (StatuteIndex: exact-citation resolution, BM25 with title bonus and synonym
// expansion, guidance appendix with relevance floor, coverage gate). Data and
// constants come from data.json, exported by scripts/39_build_static_demo.py, so
// the two implementations cannot drift on data; scripts/39 --check verifies
// that they agree on real questions.

const TOKEN_RE = /[a-z0-9ऀ-ॿ]+/g;
const NUMBER_RE = /(\d+[A-Za-z]{0,2})/;
const CITATION_RE = /(?:\b(?:Sections?|Sec\.?|Articles?|Art\.?|dhara|धारा|अनुच्छेद)\s+|§\s*)\d+[A-Za-z]{0,2}\b(?:\(\w+\))*/gi;

function tokens(text) {
  return (text.toLowerCase().match(TOKEN_RE)) || [];
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isAscii(s) {
  return /^[\x00-\x7f]*$/.test(s);
}

// Latin-script phrases need word boundaries; Devanagari must not use \b
// (combining vowels break \w-based boundaries) — plain substring works there.
function aliasMatcher(variant) {
  return isAscii(variant) ? new RegExp("\\b" + escapeRe(variant) + "\\b") : null;
}

export function buildIndex(data) {
  const P = data.params;
  const rows = data.rows;
  const byKey = new Map();
  rows.forEach((r, i) => byKey.set(`${r.act_id}:${r.section.toUpperCase()}`, i));

  const oldToNew = new Map();
  for (const m of data.mappings) {
    const newActId = data.family_to_act_id[m.new_act.toLowerCase()];
    if (!newActId) continue;
    const k = `${m.old_act.toLowerCase()}|${m.old_section.toUpperCase()}`;
    if (!oldToNew.has(k)) oldToNew.set(k, []);
    oldToNew.get(k).push(`${newActId}:${m.new_section.toUpperCase()}`);
  }

  const docTokens = rows.map(r => tokens(`${r.act_name} ${r.title || ""} ${r.text || ""} ${r.punishment_summary || ""} ${(r.tags || []).join(" ")}`));
  const titleTokens = rows.map(r => new Set([...tokens(r.title || ""), ...tokens((r.tags || []).join(" "))]));
  const docLen = docTokens.map(t => t.length);
  const avgLen = docLen.reduce((a, b) => a + b, 0) / Math.max(1, docLen.length);
  const tf = docTokens.map(t => { const m = new Map(); for (const w of t) m.set(w, (m.get(w) || 0) + 1); return m; });
  const df = new Map();
  for (const t of docTokens) for (const w of new Set(t)) df.set(w, (df.get(w) || 0) + 1);
  const n = rows.length;
  const idf = new Map();
  for (const [w, c] of df) idf.set(w, Math.log(1 + (n - c + 0.5) / (c + 0.5)));

  const synonymPatterns = Object.entries(data.synonyms).map(([phrase, exp]) => ({
    re: isAscii(phrase[0]) ? new RegExp("\\b" + escapeRe(phrase) + "\\b") : null, phrase, exp }));

  function expandQuery(query) {
    const q = query.toLowerCase();
    const extra = [];
    for (const s of synonymPatterns) if (s.re ? s.re.test(q) : q.includes(s.phrase)) extra.push(s.exp);
    return extra.length ? q + " " + extra.join(" ") : q;
  }

  function familiesIn(queryLower) {
    const fams = [];
    for (const [family, variants] of Object.entries(data.act_aliases)) {
      if (variants.some(v => { const re = aliasMatcher(v); return re ? re.test(queryLower) : queryLower.includes(v); })) fams.push(family);
    }
    return fams;
  }

  function pickByContent(candidates, query) {
    const qTokens = new Set(tokens(expandQuery(query)));
    const scored = candidates.map(key => {
      const i = byKey.get(key); let s = 0;
      for (const w of qTokens) if (tf[i].has(w)) s += idf.get(w) || 0;
      return [s, key];
    }).sort((a, b) => b[0] - a[0] || (a[1] < b[1] ? 1 : -1));
    if (scored[0][0] > 0 && (scored.length === 1 || scored[0][0] > scored[1][0])) return [scored[0][1]];
    return candidates;
  }

  function referencedKeys(query) {
    const queryLower = query.toLowerCase();
    const families = familiesIn(queryLower);
    let keys = [];
    for (const m of query.matchAll(CITATION_RE)) {
      const num = m[0].match(NUMBER_RE);
      if (!num) continue;
      const section = num[1].toUpperCase();
      const marker = m[0].toLowerCase();
      const articleLike = marker.includes("art") || marker.includes("अनुच्छेद");
      const matchFamilies = families.length ? families : (articleLike ? ["constitution"] : Object.keys(data.family_to_act_id));
      for (const family of matchFamilies) {
        const mapped = data.old_to_new_act[family];
        if (mapped) keys.push(...(oldToNew.get(`${family}|${section}`) || []));
        const actId = data.family_to_act_id[family];
        if (actId && byKey.has(`${actId}:${section}`)) keys.push(`${actId}:${section}`);
      }
      if (!families.length && !articleLike) {
        let candidates = keys.filter(k => k.endsWith(":" + section));
        if (candidates.length > 1) candidates = pickByContent(candidates, query);
        if (candidates.length !== 1) keys = keys.filter(k => !k.endsWith(":" + section));
        else { const keep = candidates[0]; keys = keys.filter(k => k === keep || !k.endsWith(":" + section)); }
      }
    }
    const seen = new Set(); const ordered = [];
    for (const k of keys) if (!seen.has(k)) { seen.add(k); ordered.push(k); }
    return ordered;
  }

  function bm25(query) {
    const qTokens = tokens(expandQuery(query));
    const scores = [];
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (const w of qTokens) {
        const f = tf[i].get(w);
        if (!f) continue;
        const w_idf = idf.get(w) || 0;
        const denom = f + P.k1 * (1 - P.b + P.b * docLen[i] / avgLen);
        s += w_idf * f * (P.k1 + 1) / denom;
        if (titleTokens[i].has(w)) s += P.title_bonus * w_idf;
      }
      if (s > 0) scores.push([s, i]);
    }
    // Python sorts (score, index) tuples descending: ties resolve to the larger index first.
    scores.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
    return scores;
  }

  function retrieve(query, k = 4) {
    const picked = [];
    for (const key of referencedKeys(query)) { picked.push(rows[byKey.get(key)]); if (picked.length >= k) break; }
    const chosen = new Set(picked.map(r => `${r.act_id}:${r.section.toUpperCase()}`));
    const bm = bm25(query);
    const scoreOf = new Map(bm.map(([s, i]) => [i, s]));
    const statutes = [], guidance = [];
    for (const [, i] of bm) {
      const row = rows[i];
      if (chosen.has(`${row.act_id}:${row.section.toUpperCase()}`)) continue;
      (row.act_id === "procedures_kb" ? guidance : statutes).push([i, row]);
    }
    const topStatute = statutes.reduce((m, [i]) => Math.max(m, scoreOf.get(i) || 0), 0);
    const floor = P.guidance_floor_ratio * topStatute;
    const statuteRows = statutes.map(([, r]) => r);
    const guidanceRows = guidance.filter(([i]) => (scoreOf.get(i) || 0) >= floor).map(([, r]) => r);
    const slots = Math.max(0, k - picked.length);
    const statuteTake = statuteRows.slice(0, slots);
    const guidancePool = guidanceRows.slice(0, P.kb_slots);
    const guidanceTake = (picked.length || statuteTake.length) ? guidancePool : guidance.map(([, r]) => r).slice(0, k);
    return [...picked, ...statuteTake, ...guidanceTake];
  }

  function coverage(query) {
    if (referencedKeys(query).length) return { top_statute_score: null, covered: true, reason: "explicit citation" };
    let best = 0;
    for (const [s, i] of bm25(query)) { if (rows[i].act_id !== "procedures_kb") { best = s; break; } }
    const covered = best >= P.coverage_min_score;
    return { top_statute_score: Math.round(best * 1000) / 1000, covered,
             reason: covered ? "bm25" : "no act in the database scores high enough" };
  }

  return { rows, retrieve, coverage, referencedKeys, params: P };
}
