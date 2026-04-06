def client_view(llm):
    gt_tokens = llm.get("GT_Tokens", 0)
    exact = llm.get("Exact_Count", 0)
    fuzzy = llm.get("Fuzzy_Count", 0)

    matched = exact + fuzzy
    pct = round((matched / gt_tokens) * 100, 2) if gt_tokens else 0.0
    missing_words = llm.get("Del_Words", []) or []
    missing_count = len(missing_words)
    missing_pct = round((missing_count / gt_tokens) * 100, 2) if gt_tokens else 0.0

    gt_names = llm.get("GT_Names", []) or []
    asr_names = llm.get("ASR_Names", []) or []
    gt_numbers = llm.get("GT_Numbers", []) or []
    asr_numbers = llm.get("ASR_Numbers", []) or []

    return {
        "Filename": llm["Filename"],
        "GT_Translit": llm["gt_translit"],
        "ASR_Translit": llm["asr_translit"],
        "Intent_GT": llm.get("Intent_GT", ""),
        "Intent_ASR": llm.get("Intent_ASR", ""),
        "Intent_Similarity_Score": llm.get("Intent_Similarity_Score", 0.0),
        "GT_Tokens": gt_tokens,
        "GT_Names": ", ".join(gt_names),
        "ASR_Names": ", ".join(asr_names),
        "GT_Names_Count": len(gt_names),
        "ASR_Names_Count": len(asr_names),
        "GT_Numbers": ", ".join(gt_numbers),
        "ASR_Numbers": ", ".join(asr_numbers),
        "GT_Numbers_Count": len(gt_numbers),
        "ASR_Numbers_Count": len(asr_numbers),
        "Exact_Words": ", ".join(llm.get("Exact_Words", [])),
        "Fuzzy_Words": ", ".join(llm.get("Fuzzy_Words", [])),
        "Exact_Count": exact,
        "Fuzzy_Count": fuzzy,
        "Matched_Count": matched,
        "Matched_Ground_Truth(%)": pct,
        "Missing_Words": ", ".join(missing_words),
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_pct,
    }


def internal_view(llm):
    gt_names = llm.get("GT_Names", []) or []
    asr_names = llm.get("ASR_Names", []) or []
    gt_numbers = llm.get("GT_Numbers", []) or []
    asr_numbers = llm.get("ASR_Numbers", []) or []

    return {
        "Filename": llm["Filename"],
        "GT_Translit": llm["gt_translit"],
        "ASR_Translit": llm["asr_translit"],
        "Intent_GT": llm.get("Intent_GT", ""),
        "Intent_ASR": llm.get("Intent_ASR", ""),
        "Intent_Similarity_Score": llm.get("Intent_Similarity_Score", 0.0),
        "GT_Tokens": llm.get("GT_Tokens", 0),
        "GT_Names": ", ".join(gt_names),
        "ASR_Names": ", ".join(asr_names),
        "GT_Names_Count": len(gt_names),
        "ASR_Names_Count": len(asr_names),
        "GT_Numbers": ", ".join(gt_numbers),
        "ASR_Numbers": ", ".join(asr_numbers),
        "GT_Numbers_Count": len(gt_numbers),
        "ASR_Numbers_Count": len(asr_numbers),
        "Subs_Words": ", ".join(llm.get("Subs_Words", [])),
        "Dels_Words": ", ".join(llm.get("Del_Words", [])),
        "Ins_Words": ", ".join(llm.get("Ins_Words", [])),
        "Subs_Count": llm.get("Subs_Count", 0),
        "Dels_Count": llm.get("Del_Count", 0),
        "Ins_Count": llm.get("Ins_Count", 0),
        "WER(%)": round(llm.get("WER", 0) * 100, 2)
    }
