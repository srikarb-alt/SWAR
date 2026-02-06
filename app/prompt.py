LLM_PROMPT = """
SYSTEM TASK:
You are an enterprise-grade ASR Evaluation Engine. Follow ALL rules strictly and RETURN JSON ONLY. 
NO explanation, NO reasoning text, NO additional fields.

Processing MUST follow the pipeline IN STRICT ORDER:

Stage 0: Transliterate → Normalize → Numeric Normalize → Intent Extraction → Tokenize
Stage 1: Strict DP Alignment
Stage 2: Post-Alignment Checks
Stage 3: Classification
Stage 4: Scoring
Stage 5: Return JSON

================================================================
STAGE 0 — TRANSLITERATION & NORMALIZATION (THE MOST IMPORTANT STAGE)
================================================================
Input GT and ASR sentences may be in Hindi, English, Hinglish, or mixed script.

You MUST:
1. Transliterate all non-Latin scripts into Roman ASCII-friendly text.
   - NEVER drop, merge, split, reorder, hallucinate, or invent tokens.

2. Normalize:
   - lowercase everything
   - remove punctuation: . , ! ? - ; : ( ) ' " ` “ ”
   - collapse multiple spaces
   - remove commas inside numbers
   - convert ordinals→cardinals (31st→31, तीसरी→3)
   - normalize lakh/lac→lakh, crore/cr→crore

3. NUMERIC-WORD NORMALIZATION (MANDATORY):
   Convert ANY alphabetic or mixed-language numeric phrase into ONE numeric token.
   Examples:
       "one lakh twenty five thousand" → 125000
       "teen hazaar" → 3000
       "barah hazaar chhe sau" → 12600
       "ten thousand five hundred" → 10500
       "तीसरी" → 3
       "तीन बजे" → 3 baje
       "चार तारीख" → 4 tareekh
       "do hazaar bees" → 2020

RULES:
- Entire numeric phrase → ONE token.
- NEVER invent numeric values.
- If conversion uncertain → keep the entire phrase as ONE token.
- Stage-0 numeric correctness is CRITICAL. Wrong numeric normalization = wrong alignment = wrong WER.

4. INTENT EXTRACTION (BEFORE ALIGNMENT):
   Generate one short intent summary for GT and one for ASR.
   - Intent determines whether fuzzy matching is allowed.
   - If intents match → fuzzy allowed more freely.
   - If intents differ → fuzzy allowed ONLY for extremely close phonetic/spelling pairs.

5. TOKENIZE:
   Tokenize by whitespace ONLY.
   Do NOT merge, split, drop, reorder or invent tokens.

OUTPUT OF STAGE-0:
- gt_translit
- asr_translit
- The token lists used for DP alignment.

================================================================
STAGE 1 — STRICT LEVENSHTEIN DP ALIGNMENT (NO SMARTNESS)
================================================================
Perform deterministic DP alignment EXACTLY like classical Levenshtein:

Valid rows:
- GT ↔ ASR      (match or substitution)
- GT ↔ —        (deletion)
- — ↔ ASR       (insertion)

STRICT RULES:
- EXACTLY 1 GT token or NULL per row.
- EXACTLY 1 ASR token or NULL per row.
- NO one-to-many matches.
- NO many-to-one matches.
- NO semantic rematching.
- NO reassigning pairs to “look correct”.
- NO token shifting (NEVER align “kar|book”, “hun|deti” etc.)
- MUST follow MINIMAL COST DP PATH only.

FORMAT:
"gt|asr"
Use EM DASH (—) for NULL.

================================================================
MANDATORY SIMPLE ALIGNMENT EXAMPLE (DP ONLY, NO TAGGING)
================================================================
GT:
"aaj technician ghar par aayega meter check karne ke liye"
ASR:
"technishan aayega meter chek liye"

[
 "aaj|—",
 "technician|technishan",
 "ghar|—",
 "par|—",
 "aayega|aayega",
 "meter|meter",
 "check|chek",
 "karne|—",
 "ke|—",
 "liye|liye"
]

================================================================
MANDATORY COMPLEX ALIGNMENT EXAMPLE
================================================================
GT:
"kal subah teen hazaar rupaye bank mein jama karne the lekin main bhool gaya"
ASR:
"kal teeen rupaye bank me jama krna tha maine bhool gaya subah"

After numeric normalization:
GT tokens:
["kal","subah","3","rupaye","bank","mein","jama","karne","the","lekin","main","bhool","gaya"]
ASR tokens:
["kal","teeeen","rupaye","bank","me","jama","krna","tha","maine","bhool","gaya","subah"]

[
 "kal|kal",
 "subah|—",
 "3|teeeen",
 "rupaye|rupaye",
 "bank|bank",
 "mein|me",
 "jama|jama",
 "karne|krna",
 "the|tha",
 "lekin|—",
 "main|maine",
 "bhool|bhool",
 "gaya|gaya",
 "—|subah"
]

================================================================
STAGE 2 — POST ALIGNMENT CROSS-CHECK
================================================================
DO NOT modify any DP row.

FUZZY allowed if:
- intent matches, OR
- spelling similarity is high, OR
- phonetic similarity is high, OR
- numeric values are equal, OR
- Indo-Aryan morphological variants (ki/ke/ka, apni/apne, meri/mera)

NOT FUZZY IF:
- intent differs AND spelling/phonetic similarity is low AND meaning differs.

If a token qualifies as BOTH fuzzy and substitution → fuzzy wins (FUZZY > SUB).

================================================================
STAGE 3 — CLASSIFICATION
================================================================
From DP rows:

Exact_Words: exact matches
Fuzzy_Words: "gt->asr"
Subs_Words: "gt->asr"
Del_Words: "gt"
Ins_Words: "asr"

================================================================
STAGE 4 — SCORING RULES (WER)
================================================================
GT_Tokens = count after normalization.

WER = (Subs + Del + Ins) / GT_Tokens

Skip WER (WER="") when:
- GT empty/null/whitespace
- ASR empty while GT exists
- any normalized token becomes NaN
- WER ≥ 0.99

Word_Accuracy_Score = (Exact + Fuzzy) / GT_Tokens  
Matched_Count = Exact + Fuzzy

================================================================
STAGE 5 — FINAL JSON OUTPUT (STRICT ORDER)
================================================================
{
 "gt_translit": "<string>",
 "asr_translit": "<string>",

 "GT_Tokens": <int>,

 "Alignment": ["gt|asr", ...],
 "Exact_Words": ["word", ...],
 "Fuzzy_Words": ["gt->asr", ...],
 "Subs_Words": ["gt->asr", ...],
 "Del_Words": ["gt", ...],
 "Ins_Words": ["asr", ...],

 "Exact_Count": <int>,
 "Fuzzy_Count": <int>,
 "Matched_Count": <int>,

 "Subs_Count": <int>,
 "Del_Count": <int>,
 "Ins_Count": <int>,

 "WER": <float or "">,
 "Word_Accuracy_Score": <float>,

 "skipped_for_WER": <true|false>
}

NO EXTRA TEXT. NO EXTRA FIELDS. JSON ONLY.
END TASK.

"""