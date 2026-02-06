def client_view(llm):
    gt_tokens = llm.get("GT_Tokens", 0)
    exact = llm.get("Exact_Count", 0)
    fuzzy = llm.get("Fuzzy_Count", 0)

    matched = exact + fuzzy
    pct = round((matched / gt_tokens) * 100, 2) if gt_tokens else 0.0

    return {
        "Filename": llm["Filename"],
        "GT_Translit": llm["gt_translit"],
        "ASR_Translit": llm["asr_translit"],
        "GT_Tokens": gt_tokens,
        "Exact_Words": ", ".join(llm.get("Exact_Words", [])),
        "Fuzzy_Words": ", ".join(llm.get("Fuzzy_Words", [])),
        "Exact_Count": exact,
        "Fuzzy_Count": fuzzy,
        "Matched_Count": matched,
        "Matched_Ground_Truth(%)": pct
    }


def internal_view(llm):
    return {
        "Filename": llm["Filename"],
        "GT_Translit": llm["gt_translit"],
        "ASR_Translit": llm["asr_translit"],
        "GT_Tokens": llm.get("GT_Tokens", 0),
        "Subs_Words": ", ".join(llm.get("Subs_Words", [])),
        "Dels_Words": ", ".join(llm.get("Del_Words", [])),
        "Ins_Words": ", ".join(llm.get("Ins_Words", [])),
        "Subs_Count": llm.get("Subs_Count", 0),
        "Dels_Count": llm.get("Del_Count", 0),
        "Ins_Count": llm.get("Ins_Count", 0),
        "WER(%)": round(llm.get("WER", 0) * 100, 2)
    }
