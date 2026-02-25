"""
Groq API client for intervention script generation.
"""
import math
import os
import re
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

# Use frontend config to avoid clashing with root config
from frontend.config import GROQ_API_KEY


SYSTEM_PROMPT = """You are an Anti-Fraud Intervention Specialist. Write a CONCISE phone script (MAX 200 words) for each student.

=== ABSOLUTE RULES ===
1. STRICT 200 WORD LIMIT - be direct and concise.
2. ONLY use the SPECIFIC_ACTIONS provided - no generic advice.
3. Reference SPECIFIC numbers from their profile.

=== SCRIPT STRUCTURE (Keep it short) ===
OPENING (1 sentence): State their EXACT risk with specific numbers.
DANGER (1 sentence): What happens if they don't act.
ACTIONS (2 bullet points): Use ONLY the SPECIFIC_ACTIONS provided.
CLOSING (1 sentence): One specific next step.

=== FORBIDDEN PHRASES ===
- "Download the Scameter app"
- "Call 18222 hotline"
- "Report to police"
- "Stay vigilant" / "Be careful"
- Generic safety tips

Output: Concise script only. MAX 200 words. Every word must count."""

USER_PROMPT_TEMPLATE = """=== STUDENT PROFILE ===
Risk Tier: {risk_tier} | Score: {risk_score}/100 | Age: {age}

=== PRIMARY SCENARIO: {scenario_name} ===
{scenario_description}

=== TOP RISK SIGNALS (mention these with EXACT numbers) ===
{top_risk_signals}

=== WHY THIS IS DANGEROUS ===
{danger_explanation}

=== SPECIFIC_ACTIONS (USE ONLY THESE - DO NOT ADD GENERIC ADVICE) ===
{specific_actions}

=== DATA POINTS TO REFERENCE ===
- Behavior Score: {behavior_score}/100 {behavior_interpretation}
- Exposure Score: {exposure_score}/100 {exposure_interpretation}
- Mainland Calls: {mainland_cnt} | Mainland-to-HK: {mainland_to_hk_cnt}
- Total Messages: {total_msg_cnt} | App Peak: {app_max_cnt}
- Fraud Number Contact: {fraud_msisdn_present}

=== YOUR TASK ===
Write a CONCISE intervention script (MAX 200 words) using ONLY the SPECIFIC_ACTIONS above.
Include key numbers from the data points. NO generic suggestions.
Be direct and actionable - every word must count."""


def _split_risk_reasons(text):
    if not text:
        return []
    return [part.strip() for part in str(text).split('|') if part.strip()]


def _as_int(value, default=0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return int(parsed)


RISK_REASON_ACTIONS = [
    {
        "key": "confirmed_fraud_contact",
        "patterns": ["contact from confirmed fraud number", "confirmed fraud number", "fraud number", "blacklist"],
        "actions": [
            "Block the number and report it; keep call logs or screenshots.",
            "If any details were shared, change passwords and alert your bank or e-wallet."
        ],
    },
    {
        "key": "answered_suspected_calls",
        "patterns": ["answered suspected fraud calls", "partial engagement"],
        "actions": [
            "Stop contact immediately and do not send money or codes.",
            "Monitor recent transactions and notify your bank or e-wallet if anything looks unusual."
        ],
    },
    {
        "key": "called_back_suspected_number",
        "patterns": ["called back suspected fraud number", "active engagement doubles risk"],
        "actions": [
            "Stop calling back and block the number; keep a record of the calls.",
            "If any payments were made, contact your bank or e-wallet provider right away."
        ],
    },
    {
        "key": "replied_to_suspected_messages",
        "patterns": ["replied to suspected fraud messages", "engagement increases exposure"],
        "actions": [
            "Do not reply further or tap links in messages; block and report the sender.",
            "Check suspicious texts with Scameter before taking any action."
        ],
    },
    {
        "key": "behavior_multiplier",
        "patterns": ["behavior multiplier"],
        "actions": [
            "Because engagement indicators are elevated, pause contact and review recent calls or messages.",
            "If you shared any personal details, reset passwords and enable two-factor authentication."
        ],
    },
    {
        "key": "impersonation_or_authority",
        "patterns": ["impersonation", "authority threats", "police", "immigration", "customs", "spoofed authority threats"],
        "actions": [
            "Officials do not request payments or OTPs by phone; hang up and verify via official hotlines.",
            "Do not install remote-control apps or share personal details with callers."
        ],
    },
    {
        "key": "young_student_profile",
        "patterns": ["young student profile"],
        "actions": [
            "If you feel pressured, pause and verify with a trusted staff member or family member.",
            "Avoid sharing personal details or IDs on a first contact; verify the caller first."
        ],
    },
    {
        "key": "mainland_or_cross_border",
        "patterns": ["some mainland calls", "cross-border scam traffic", "mainland", "cross-border"],
        "actions": [
            "Do not return unknown cross-border calls; verify any claim through official numbers.",
            "Avoid sharing ID details over the phone with overseas callers."
        ],
    },
    {
        "key": "high_mainland_call_volume",
        "patterns": ["high mainland call volume", "telecom scam outreach"],
        "actions": [
            "Treat telecom-impersonation claims as suspicious; verify line or SIM issues via official channels.",
            "Use call screening or unknown-number blocking if you are receiving heavy volumes."
        ],
    },
    {
        "key": "overseas_unknown_calls",
        "patterns": ["high overseas unknown calls", "overseas unknown calls", "overseas"],
        "actions": [
            "Treat overseas caller IDs as unverified; hang up and call back using trusted numbers.",
            "If pressured, pause the conversation and contact a trusted staff member."
        ],
    },
    {
        "key": "job_scam",
        "patterns": ["online job", "upfront payments"],
        "actions": [
            "Legitimate jobs do not ask for upfront fees; verify employer details before sharing data.",
            "Avoid moving the conversation to private channels if the offer is vague or rushed."
        ],
    },
    {
        "key": "investment_scam",
        "patterns": ["investment", "crypto", "fx"],
        "actions": [
            "Be wary of guaranteed returns; verify licensing before investing.",
            "Never share wallet seed phrases or transfer money under time pressure."
        ],
    },
    {
        "key": "romance_scam",
        "patterns": ["romance", "trust-building"],
        "actions": [
            "Avoid sending money to online acquaintances, even for emergencies.",
            "Verify identity through independent channels and talk to someone you trust."
        ],
    },
]


def _dedupe_keep_order(items):
    seen = set()
    deduped = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _actions_from_risk_reasons(reasons):
    reason_text = " | ".join(reasons)
    actions = []
    for entry in RISK_REASON_ACTIONS:
        if any(pattern in reason_text for pattern in entry["patterns"]):
            actions.extend(entry["actions"])
    return actions


def _add_signal(signals, key, label, score):
    current = signals.get(key)
    if current is None or score > current["score"]:
        signals[key] = {"label": label, "score": score}


RISK_SIGNAL_PATTERNS = [
    (
        "confirmed_fraud_contact",
        re.compile(r"contact from confirmed fraud number", re.IGNORECASE),
        lambda m: ("Contact with a confirmed fraud number", 90),
    ),
    (
        "answered_suspected_calls",
        re.compile(r"answered suspected fraud calls (\\d+)x", re.IGNORECASE),
        lambda m: (f"Answered suspected fraud calls ({m.group(1)}x)", 60 + 5 * int(m.group(1))),
    ),
    (
        "called_back_suspected_number",
        re.compile(r"called back suspected fraud number (\\d+)x", re.IGNORECASE),
        lambda m: (f"Called back suspected fraud number ({m.group(1)}x)", 70 + 6 * int(m.group(1))),
    ),
    (
        "replied_to_suspected_messages",
        re.compile(r"replied to suspected fraud messages (\\d+)x", re.IGNORECASE),
        lambda m: (f"Replied to suspected fraud messages ({m.group(1)}x)", 55 + 5 * int(m.group(1))),
    ),
    (
        "behavior_multiplier",
        re.compile(r"behavior multiplier: x([0-9.]+)", re.IGNORECASE),
        lambda m: (f"Behavior multiplier x{m.group(1)}", 50 + int(float(m.group(1)) * 10)),
    ),
    (
        "high_overseas_unknown_calls",
        re.compile(r"high overseas unknown calls \\((\\d+)\\)", re.IGNORECASE),
        lambda m: (f"High overseas unknown calls ({m.group(1)})", 50 + int(m.group(1))),
    ),
    (
        "overseas_unknown_calls",
        re.compile(r"overseas unknown calls \\((\\d+)\\)", re.IGNORECASE),
        lambda m: (f"Overseas unknown calls ({m.group(1)})", 35 + int(m.group(1))),
    ),
    (
        "high_mainland_call_volume",
        re.compile(r"high mainland call volume \\((\\d+)\\)", re.IGNORECASE),
        lambda m: (f"High Mainland call volume ({m.group(1)})", 50 + int(m.group(1))),
    ),
    (
        "some_mainland_calls",
        re.compile(r"some mainland calls \\((\\d+)\\)", re.IGNORECASE),
        lambda m: (f"Mainland calls ({m.group(1)})", 25 + int(m.group(1))),
    ),
    (
        "young_student_profile",
        re.compile(r"young student profile", re.IGNORECASE),
        lambda m: ("Young student profile targeted by impersonation scams", 20),
    ),
]


def _extract_risk_signals_from_reasons(reasons):
    signals = {}
    for reason in reasons:
        for key, pattern, builder in RISK_SIGNAL_PATTERNS:
            match = pattern.search(reason)
            if not match:
                continue
            label, score = builder(match)
            _add_signal(signals, key, label, score)
    return signals


def _extract_numeric_signals(profile):
    signals = {}
    total_msg_cnt = _as_int(profile.get('total_msg_cnt', 0))
    total_voice_cnt = _as_int(profile.get('total_voice_cnt', 0))
    mainland_to_hk_cnt = _as_int(profile.get('mainland_to_hk_cnt', 0))
    mainland_cnt = _as_int(profile.get('mainland_cnt', 0))
    from_china_mobile_call_cnt = _as_int(profile.get('from_china_mobile_call_cnt', 0))
    app_max_cnt = _as_int(profile.get('app_max_cnt', 0))

    if total_msg_cnt >= 13:
        _add_signal(signals, "high_message_volume", f"High message volume ({total_msg_cnt})", 25 + total_msg_cnt)
    if total_voice_cnt >= 17:
        _add_signal(signals, "high_call_volume", f"High call volume ({total_voice_cnt})", 25 + total_voice_cnt)
    if mainland_to_hk_cnt >= 6:
        _add_signal(signals, "mainland_to_hk_calls", f"Mainland-to-HK calls ({mainland_to_hk_cnt})", 35 + mainland_to_hk_cnt)
    if mainland_cnt >= 11:
        _add_signal(signals, "mainland_calls", f"Mainland call count ({mainland_cnt})", 30 + mainland_cnt)
    if from_china_mobile_call_cnt >= 1:
        _add_signal(signals, "china_mobile_calls", f"Calls from China Mobile ({from_china_mobile_call_cnt})", 30 + from_china_mobile_call_cnt)
    if app_max_cnt >= 22:
        _add_signal(signals, "high_app_activity", f"High app activity (peak {app_max_cnt})", 20 + app_max_cnt)

    return signals


def _top_risk_signals(profile, max_items=3):
    reasons = _split_risk_reasons(profile.get('risk_reason', ''))
    signals = _extract_risk_signals_from_reasons(reasons)
    numeric_signals = _extract_numeric_signals(profile)

    for key, data in numeric_signals.items():
        _add_signal(signals, key, data["label"], data["score"])

    ranked = sorted(signals.values(), key=lambda item: item["score"], reverse=True)
    return [item["label"] for item in ranked[:max_items]]


def _determine_primary_scenario(profile):
    """
    Determine the PRIMARY risk scenario for this student.
    Returns a tuple: (scenario_name, scenario_description)
    """
    reasons = " | ".join(_split_risk_reasons(profile.get('risk_reason', ''))).lower()
    behavior_score = _as_int(profile.get('behavior_score', 0))
    exposure_score = _as_int(profile.get('exposure_score', 0))
    fraud_flag = str(profile.get('fraud_msisdn_present', '')).strip().lower() in ('true', '1', 'yes')
    mainland_to_hk_cnt = _as_int(profile.get('mainland_to_hk_cnt', 0))
    mainland_cnt = _as_int(profile.get('mainland_cnt', 0))
    from_china_mobile_call_cnt = _as_int(profile.get('from_china_mobile_call_cnt', 0))
    total_msg_cnt = _as_int(profile.get('total_msg_cnt', 0))
    app_max_cnt = _as_int(profile.get('app_max_cnt', 0))
    
    # Priority 1: Confirmed fraud contact (highest danger)
    if fraud_flag or 'confirmed fraud' in reasons or 'fraud number' in reasons or 'blacklist' in reasons:
        return ("CONFIRMED_FRAUD_CONTACT", 
                "Student has had DIRECT CONTACT with a CONFIRMED fraud number. This is the highest risk - they may already be a victim or actively being targeted. The fraudster has their number and may call again.")
    
    # Priority 2: Active engagement (called back, answered, replied)
    if 'called back' in reasons or 'answered suspected' in reasons or 'replied to suspected' in reasons or behavior_score >= 70:
        engagement_type = []
        if 'called back' in reasons:
            engagement_type.append("called back suspected numbers")
        if 'answered suspected' in reasons:
            engagement_type.append("answered suspected fraud calls")
        if 'replied to suspected' in reasons:
            engagement_type.append("replied to suspected fraud messages")
        engagement_str = ", ".join(engagement_type) if engagement_type else "showed engagement with suspicious contacts"
        return ("ACTIVE_ENGAGEMENT",
                f"Student has ACTIVELY ENGAGED with potential fraudsters - they {engagement_str}. This indicates they are being groomed and may be at the stage of trusting the scammer.")
    
    # Priority 3: Cross-border/Mainland risk pattern
    if mainland_to_hk_cnt >= 6 or mainland_cnt >= 11 or from_china_mobile_call_cnt >= 3 or 'mainland' in reasons or 'cross-border' in reasons:
        return ("CROSS_BORDER_RISK",
                f"Student shows significant CROSS-BORDER call patterns (Mainland calls: {mainland_cnt}, Mainland-to-HK: {mainland_to_hk_cnt}). This pattern is common in impersonation scams where fraudsters pose as Mainland police, customs, or telecom officials.")
    
    # Priority 4: High exposure (many suspicious contacts but no engagement yet)
    if exposure_score >= 60 or 'overseas unknown calls' in reasons:
        return ("HIGH_EXPOSURE",
                f"Student has HIGH EXPOSURE to suspicious numbers (Exposure Score: {exposure_score}/100) but has not engaged yet. They are being heavily targeted and one answered call could start the scam process.")
    
    # Priority 5: Digital vulnerability (high app/message activity)
    if total_msg_cnt >= 15 or app_max_cnt >= 25:
        return ("DIGITAL_VULNERABILITY",
                f"Student shows HIGH DIGITAL ACTIVITY (Messages: {total_msg_cnt}, App Peak: {app_max_cnt}). This makes them vulnerable to phishing links, malicious apps, and online job/investment scams.")
    
    # Default: General vulnerability
    return ("GENERAL_VULNERABILITY",
            "Student shows elevated risk signals that require attention. While no single factor is critical, the combination of factors warrants preventive intervention.")


def _generate_danger_explanation(profile, scenario_name):
    """
    Generate a specific explanation of WHY this student is in danger.
    """
    risk_tier = profile.get('risk_tier', 'UNKNOWN')
    reasons = _split_risk_reasons(profile.get('risk_reason', ''))
    behavior_score = _as_int(profile.get('behavior_score', 0))
    exposure_score = _as_int(profile.get('exposure_score', 0))
    identity_score = _as_int(profile.get('identity_score', 0))
    
    explanations = []
    
    if scenario_name == "CONFIRMED_FRAUD_CONTACT":
        explanations.append("The number that contacted this student is on our confirmed fraud blacklist - it has been used in previous scam cases.")
        explanations.append("Fraudsters often call back victims multiple times, escalating pressure or pretending to be different officials.")
        if behavior_score > 0:
            explanations.append(f"Concerning: This student has shown engagement (Behavior Score: {behavior_score}), meaning they may have already shared information.")
    
    elif scenario_name == "ACTIVE_ENGAGEMENT":
        explanations.append("By engaging with the caller, the student has signaled they are receptive - fraudsters will now intensify contact.")
        explanations.append("Common pattern: First call builds trust, second call creates urgency, third call demands money or personal info.")
        if 'called back' in " ".join(reasons).lower():
            explanations.append("Calling back is especially dangerous - it confirms the student's interest and may incur premium call charges.")
    
    elif scenario_name == "CROSS_BORDER_RISK":
        explanations.append("Cross-border call patterns are a red flag for impersonation scams targeting HK students.")
        explanations.append("Typical script: Caller claims to be Mainland police/customs, says student's identity was used for crimes, demands 'verification' or 'bail' money.")
        explanations.append("These scammers speak Mandarin and use official-sounding titles to intimidate young students.")
    
    elif scenario_name == "HIGH_EXPOSURE":
        explanations.append("High exposure means this student's number is circulating among scam networks.")
        explanations.append("They will likely receive more calls - each unanswered call is a win, but one mistake could start the scam.")
    
    elif scenario_name == "DIGITAL_VULNERABILITY":
        explanations.append("High digital activity increases attack surface for phishing and social engineering.")
        explanations.append("Common traps: Fake job offers requiring 'training fees', investment groups promising guaranteed returns, romance scams.")
    
    return " ".join(explanations)


def _interpret_score(score, score_type):
    """Generate human-readable interpretation of a sub-score."""
    if score_type == "identity":
        if score >= 70:
            return "(HIGH - young, vulnerable profile heavily targeted by scammers)"
        elif score >= 40:
            return "(MODERATE - some vulnerability factors present)"
        else:
            return "(LOW - less vulnerable profile)"
    
    elif score_type == "exposure":
        if score >= 70:
            return "(HIGH - significant contact with suspicious numbers)"
        elif score >= 40:
            return "(MODERATE - some suspicious contact detected)"
        else:
            return "(LOW - minimal suspicious contact)"
    
    elif score_type == "behavior":
        if score >= 70:
            return "(HIGH - ACTIVE ENGAGEMENT with potential fraud - highest concern)"
        elif score >= 40:
            return "(MODERATE - some engagement indicators)"
        else:
            return "(LOW - no engagement detected)"
    
    return ""


def _trim_to_word_limit(text, max_words=None):
    """Optional word limit - if max_words is None, return full text."""
    if max_words is None:
        return text.strip()
    
    words = text.split()
    if len(words) <= max_words:
        return text.strip()

    truncated = " ".join(words[:max_words])
    # Try to end on a sentence boundary if possible.
    for sep in (".", "!", "?"):
        idx = truncated.rfind(sep)
        if idx > 0:
            truncated = truncated[:idx + 1]
            break
    return truncated.strip()


# ============================================================
# DYNAMIC SOLUTION GENERATOR - Builds unique solutions from ALL risk factors
# Each factor contributes specific advice; solutions are assembled dynamically
# ============================================================

def _build_scenario_actions(profile, scenario_name):
    """
    Dynamically generate solutions by analyzing ALL risk factors.
    Each factor contributes specific actionable advice.
    Solutions are assembled based on the unique combination of factors present.
    """
    # Extract all data points
    behavior_score = _as_int(profile.get('behavior_score', 0))
    exposure_score = _as_int(profile.get('exposure_score', 0))
    identity_score = _as_int(profile.get('identity_score', 0))
    risk_score = _as_int(profile.get('risk_score', 0))
    mainland_cnt = _as_int(profile.get('mainland_cnt', 0))
    mainland_to_hk_cnt = _as_int(profile.get('mainland_to_hk_cnt', 0))
    total_msg_cnt = _as_int(profile.get('total_msg_cnt', 0))
    total_voice_cnt = _as_int(profile.get('total_voice_cnt', 0))
    app_max_cnt = _as_int(profile.get('app_max_cnt', 0))
    from_china_mobile_call_cnt = _as_int(profile.get('from_china_mobile_call_cnt', 0))
    fraud_flag = str(profile.get('fraud_msisdn_present', '')).strip().lower() in ('true', '1', 'yes')
    risk_tier = profile.get('risk_tier', 'UNKNOWN')
    
    # Parse engagement details from risk_reason
    reasons_text = profile.get('risk_reason', '').lower()
    answered_match = re.search(r'answered.*?(\d+)x', reasons_text)
    callback_match = re.search(r'called back.*?(\d+)x', reasons_text)
    replied_match = re.search(r'replied.*?(\d+)x', reasons_text)
    answered_cnt = int(answered_match.group(1)) if answered_match else 0
    callback_cnt = int(callback_match.group(1)) if callback_match else 0
    replied_cnt = int(replied_match.group(1)) if replied_match else 0
    
    # Build a list of (priority, solution) tuples
    # Priority determines order; higher = more urgent
    solutions = []
    
    # =========================================================================
    # FACTOR 1: Confirmed fraud number contact
    # =========================================================================
    if fraud_flag or 'confirmed fraud' in reasons_text or 'blacklist' in reasons_text:
        if behavior_score >= 60:
            solutions.append((100, "URGENT: You contacted a confirmed fraud number AND engaged with them. Open your bank app RIGHT NOW and check for unauthorized transactions in the last 7 days."))
        elif behavior_score >= 30:
            solutions.append((90, "You've been contacted by a confirmed fraud number. Review your recent conversations - what information did you share? Names, addresses, ID numbers?"))
        else:
            solutions.append((80, "A confirmed fraud number reached you, but you didn't engage much. Block it now: long-press the number > Block. Watch for similar numbers calling back."))
    
    # =========================================================================
    # FACTOR 2: Called back (most dangerous active engagement)
    # =========================================================================
    if callback_cnt >= 3:
        solutions.append((98, f"CRITICAL: You called back {callback_cnt} times. Scammers now see you as a 'hot lead'. Expect intense follow-up. DO NOT answer any more calls from unknown numbers."))
        solutions.append((85, "If they gave you any 'case numbers', 'reference IDs', or officer names - write them down. These are fake but help identify the scam type if you report."))
    elif callback_cnt == 2:
        solutions.append((95, "You called back twice - this strongly signals interest to scammers. Block this number AND enable 'Silence Unknown Callers' in your phone settings."))
        solutions.append((75, "Review both callback conversations: What did they ask? What did you confirm? Each piece of info you gave makes their next approach more convincing."))
    elif callback_cnt == 1:
        solutions.append((88, "You called back once. They now know your number is active and you're willing to engage. Don't call again - even to 'cancel' or 'clarify'."))
        solutions.append((70, "For the next 2 weeks, let ALL unknown calls go to voicemail. If it's real, they'll leave a message you can verify."))
    
    # =========================================================================
    # FACTOR 3: Answered suspected fraud calls
    # =========================================================================
    if answered_cnt >= 5:
        solutions.append((92, f"You've answered {answered_cnt} suspected fraud calls. This pattern suggests ongoing 'grooming' - the scammer is building trust before the big ask."))
        solutions.append((82, "Their next call will likely request something: money, gift cards, bank login, or 'just' your ID photo. Whatever they ask - the answer is NO."))
    elif answered_cnt >= 3:
        solutions.append((85, f"You answered {answered_cnt} suspected calls. Write down what they discussed: Job offer? Investment? Legal trouble? Identifying the scam type helps you see through it."))
        solutions.append((72, "If they asked you to keep conversations 'confidential' or 'secret from family', that's a classic manipulation tactic. Tell someone you trust TODAY."))
    elif answered_cnt >= 1:
        solutions.append((75, f"You answered {answered_cnt} suspected fraud call(s). Think back: Did you confirm your name? Say 'yes' to anything? Even small confirmations help scammers."))
    
    # =========================================================================
    # FACTOR 4: Replied to suspicious messages
    # =========================================================================
    if replied_cnt >= 3:
        solutions.append((88, f"TEXT SCAM ALERT: You replied to {replied_cnt} suspicious messages. Check your browser history - did any links take you to login pages? Those may be phishing sites."))
        solutions.append((78, "If you entered ANY passwords through links in those messages, change them NOW. Use a different password for each account."))
    elif replied_cnt >= 1:
        solutions.append((72, f"You replied to {replied_cnt} suspicious message(s). Did they contain links? If you clicked any, check what sites they led to. Unfamiliar login pages = phishing."))
        solutions.append((65, "Block this sender: iPhone: hold message > Report Junk. Android: hold > Block & Report spam. Don't reply again, even to say 'stop'."))
    
    # =========================================================================
    # FACTOR 5: High behavior score (general engagement indicator)
    # =========================================================================
    if behavior_score >= 80 and not (callback_cnt > 0 or answered_cnt > 2):
        solutions.append((90, f"Your engagement score ({behavior_score}) is very high. Even if you don't remember specific calls, you've had significant interaction with risky contacts."))
        solutions.append((80, "Check your phone for any apps you were asked to install for 'verification', 'security', or 'work'. These may be spyware - delete and run a security scan."))
    elif behavior_score >= 50 and not (callback_cnt > 0 or answered_cnt > 1):
        solutions.append((70, f"Moderate engagement detected (score: {behavior_score}). Review recent calls and messages - anything that created urgency or asked for secrecy is suspicious."))
    
    # =========================================================================
    # FACTOR 6: Cross-border / Mainland calls (impersonation scam risk)
    # =========================================================================
    if mainland_to_hk_cnt >= 12 or (mainland_cnt >= 18 and from_china_mobile_call_cnt >= 5):
        solutions.append((86, "IMPERSONATION SCAM PATTERN: Your call volume matches organized fraud targeting HK students. If ANY caller claimed to be police/customs/bank - 100% scam."))
        solutions.append((76, "Real Mainland police CANNOT: call HK numbers to arrest you, demand money for 'bail', or ask you to transfer to 'safe accounts'. These are all scam scripts."))
        solutions.append((66, "If you're genuinely worried about legal issues, call HK Police directly at 2527 7177. NEVER use phone numbers a caller gives you."))
    elif mainland_to_hk_cnt >= 8:
        solutions.append((80, f"High cross-border call volume ({mainland_to_hk_cnt} Mainland-to-HK). If callers claimed to be from '公安局', '入境处', or '海关' - this is a known scam script."))
        solutions.append((68, "Real government officials send written notices for serious matters. Phone threats demanding immediate payment are ALWAYS scams."))
    elif mainland_to_hk_cnt >= 5 or mainland_cnt >= 10:
        solutions.append((65, f"You've received {mainland_to_hk_cnt} cross-border calls. Be aware: calls claiming to be from Mainland authorities are a top scam targeting HK students."))
    
    # =========================================================================
    # FACTOR 7: China Mobile specific calls
    # =========================================================================
    if from_china_mobile_call_cnt >= 3 and mainland_to_hk_cnt < 8:
        solutions.append((60, f"Calls from China Mobile detected ({from_china_mobile_call_cnt}). If they claimed 'SIM problems' or 'account issues', verify directly with your carrier - not through their number."))
    
    # =========================================================================
    # FACTOR 8: High exposure but low engagement (prevention opportunity)
    # =========================================================================
    if exposure_score >= 70 and behavior_score < 25:
        solutions.append((75, "GOOD NEWS: Your number is being targeted, but you haven't engaged. Keep it this way - your best defense is continued non-response."))
        solutions.append((68, "Enable 'Silence Unknown Callers': iPhone > Settings > Phone. Android > Settings > Blocked numbers > Block unknown callers."))
    elif exposure_score >= 50 and behavior_score < 20:
        solutions.append((55, "Your number has some exposure to suspicious contacts, but minimal engagement. Continue ignoring unknown callers - every unanswered scam call is a win."))
    
    # =========================================================================
    # FACTOR 9: High call volume (number circulating in scam networks)
    # =========================================================================
    if total_voice_cnt >= 30:
        solutions.append((72, f"Your phone received {total_voice_cnt} calls - your number is actively circulating. Consider registering on Do-Not-Call: https://hkdnc.ofca.gov.hk"))
        solutions.append((58, "Install a caller ID app (Truecaller, Whoscall) that flags known scam numbers. This helps you screen before answering."))
    elif total_voice_cnt >= 20:
        solutions.append((60, f"Elevated call volume ({total_voice_cnt}). Don't return missed calls from unknown numbers - legitimate callers leave voicemails."))
    
    # =========================================================================
    # FACTOR 10: High message volume (phishing exposure)
    # =========================================================================
    if total_msg_cnt >= 25:
        solutions.append((70, f"High message volume ({total_msg_cnt}). Before clicking ANY link in messages, check: Is the URL spelled correctly? Does the domain match the claimed sender?"))
        solutions.append((62, "Enable SMS filtering: iPhone > Settings > Messages > Filter Unknown Senders. This separates messages from unknown contacts."))
    elif total_msg_cnt >= 15:
        solutions.append((55, f"You receive many messages ({total_msg_cnt}). Be cautious with links - hover/long-press to preview URLs before clicking. Misspelled domains = scam."))
    
    # =========================================================================
    # FACTOR 11: High app activity (malware/permission risk)
    # =========================================================================
    if app_max_cnt >= 35:
        solutions.append((75, f"Very high app activity (peak: {app_max_cnt}). Check Settings > Privacy > App Permissions. Revoke access for apps that don't need contacts, SMS, or camera."))
        solutions.append((65, "Review recent app downloads. Delete anything you don't recognize or that asked for unusual permissions (screen recording, accessibility)."))
    elif app_max_cnt >= 25:
        solutions.append((62, f"Elevated app usage (peak: {app_max_cnt}). Only install apps from official stores. Never install APKs or apps via links in messages."))
    
    # =========================================================================
    # FACTOR 12: Combined app + message activity (job/investment scam risk)
    # =========================================================================
    if app_max_cnt >= 20 and total_msg_cnt >= 15:
        solutions.append((68, "With your app and message activity, watch for 'easy job' or 'investment opportunity' messages. Anything requiring upfront payment is a scam."))
    
    # =========================================================================
    # FACTOR 13: Student identity vulnerability
    # =========================================================================
    if identity_score >= 70:
        solutions.append((65, "Your student profile makes you a prime target. Common scams targeting students: fake internships, scholarship 'processing fees', tutoring 'deposits'."))
        solutions.append((55, "If anyone asks for your student ID, HKID copy, or university login credentials, verify through official university channels FIRST - not through the requester."))
    elif identity_score >= 50:
        solutions.append((50, "As a student, be extra cautious of 'too good to be true' offers: high-paying easy jobs, guaranteed investment returns, unexpected scholarships."))
    
    # =========================================================================
    # FACTOR 14: Password/credential sharing indicators
    # =========================================================================
    if 'password' in reasons_text or 'credential' in reasons_text or 'login' in reasons_text:
        solutions.append((95, "CREDENTIAL ALERT: If you shared any passwords or login details, change them IMMEDIATELY. Start with bank accounts and email."))
        solutions.append((85, "Enable 2-factor authentication on all important accounts. Even if passwords are compromised, 2FA adds a critical barrier."))
    
    # =========================================================================
    # FACTOR 15: Banking/payment mentioned
    # =========================================================================
    if 'bank' in reasons_text or 'payment' in reasons_text or 'transfer' in reasons_text:
        solutions.append((92, "Banking mentioned in your risk factors. Check your accounts for: pending transfers, new payees added, or unfamiliar transaction requests."))
    
    # =========================================================================
    # FACTOR 16: Urgency manipulation detected
    # =========================================================================
    if 'urgent' in reasons_text or 'immediate' in reasons_text:
        solutions.append((78, "Urgency is a manipulation tactic. Real emergencies from banks/government have proper channels. If someone says 'act NOW', pause instead."))
    
    # =========================================================================
    # FACTOR 17: Secrecy demands detected
    # =========================================================================
    if 'secret' in reasons_text or 'confidential' in reasons_text or "don't tell" in reasons_text:
        solutions.append((80, "Secrecy demands are a red flag. Scammers isolate victims. If anyone tells you NOT to discuss something with family/friends, do the opposite - tell someone immediately."))
    
    # =========================================================================
    # FACTOR 18: Police/authority impersonation keywords
    # =========================================================================
    if 'police' in reasons_text or 'immigration' in reasons_text or '公安' in reasons_text or 'customs' in reasons_text:
        solutions.append((82, "If anyone claimed to be police/immigration/customs demanding money or secrecy - 100% scam. Real authorities don't operate this way."))
    
    # =========================================================================
    # FACTOR 19: Investment/crypto scam keywords
    # =========================================================================
    if 'invest' in reasons_text or 'crypto' in reasons_text or 'forex' in reasons_text or 'trading' in reasons_text:
        solutions.append((77, "Investment scam indicators detected. Remember: Guaranteed returns don't exist. Anyone promising 'no risk' profits is lying."))
    
    # =========================================================================
    # FACTOR 20: Job scam keywords
    # =========================================================================
    if 'job' in reasons_text or 'work from home' in reasons_text or 'easy money' in reasons_text:
        solutions.append((73, "Job scam warning: Legitimate employers NEVER ask for deposits, 'training fees', or your bank details before you start work."))
    
    # =========================================================================
    # FACTOR 21: Combined high risk score
    # =========================================================================
    combined_risk = behavior_score + exposure_score
    if combined_risk >= 130:
        solutions.append((88, f"Your combined risk score ({combined_risk}) is in the highest bracket. Consider temporarily changing your phone number if harassment continues."))
    elif combined_risk >= 100:
        solutions.append((72, "Elevated combined risk. Share this information with a trusted family member or friend - having someone else aware helps you stay objective."))
    
    # =========================================================================
    # FALLBACK: General risk level (if few factors triggered)
    # =========================================================================
    if risk_score >= 70 and len(solutions) < 2:
        solutions.append((60, "Your overall risk score is elevated. Review recent communications for anything that created pressure, promised easy money, or demanded quick action."))
    
    if risk_score >= 40 and len(solutions) < 2:
        solutions.append((45, "Moderate risk detected. Key principle: Real organizations never ask for money, passwords, or sensitive data over unsolicited calls/messages."))
    
    # If somehow no factors triggered, add baseline advice
    if len(solutions) == 0:
        solutions.append((40, "Stay aware: If any call or message creates urgency ('act now'), demands secrecy ('don't tell anyone'), or seems too good to be true - it's likely a scam."))
        solutions.append((35, "When in doubt, verify independently: Look up the organization's official number yourself and call them - never use contact info provided by the suspicious caller."))
    
    # Sort by priority (highest first) and return top solutions
    solutions.sort(key=lambda x: x[0], reverse=True)
    return [sol[1] for sol in solutions[:5]]  # Return top 5 most relevant


def _build_action_guidance(profile):
    """
    Legacy function - now redirects to scenario-based actions.
    """
    scenario_name, _ = _determine_primary_scenario(profile)
    return _build_scenario_actions(profile, scenario_name)


@st.cache_resource
def get_groq_client():
    """Initialize Groq client with caching to avoid reconnecting."""
    if not GROQ_AVAILABLE:
        return None
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def generate_intervention_script(student_profile: dict) -> str:
    """
    Generate a PERSONALIZED intervention script for a high-risk student.
    
    Args:
        student_profile: Dict with risk_tier, risk_score, age, risk_reason, sub-scores
    
    Returns:
        Generated intervention script string
    """
    client = get_groq_client()
    
    if client is None:
        if not GROQ_AVAILABLE:
            return "⚠️ Groq library not installed. Run: `pip install groq`"
        return "⚠️ GROQ_API_KEY not configured. Set it in your .env file."
    
    # Determine primary scenario FIRST
    scenario_name, scenario_description = _determine_primary_scenario(student_profile)
    
    # Build scenario-specific actions (completely different per scenario)
    specific_actions = _build_scenario_actions(student_profile, scenario_name)
    actions_block = "\n".join(f"{i+1}. {action}" for i, action in enumerate(specific_actions))
    
    # Get top risk signals
    top_risk_signals = _top_risk_signals(student_profile)
    top_risk_block = "\n".join(f"- {item}" for item in top_risk_signals) if top_risk_signals else "None"
    
    # Generate danger explanation
    danger_explanation = _generate_danger_explanation(student_profile, scenario_name)
    
    # Generate score interpretations
    identity_score = _as_int(student_profile.get('identity_score', 0))
    exposure_score = _as_int(student_profile.get('exposure_score', 0))
    behavior_score = _as_int(student_profile.get('behavior_score', 0))

    # Format the user prompt with all personalized context
    user_prompt = USER_PROMPT_TEMPLATE.format(
        risk_tier=student_profile.get('risk_tier', 'UNKNOWN'),
        risk_score=student_profile.get('risk_score', 0),
        age=student_profile.get('age', 'Unknown'),
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        top_risk_signals=top_risk_block,
        danger_explanation=danger_explanation,
        specific_actions=actions_block,
        exposure_score=exposure_score,
        exposure_interpretation=_interpret_score(exposure_score, "exposure"),
        behavior_score=behavior_score,
        behavior_interpretation=_interpret_score(behavior_score, "behavior"),
        total_msg_cnt=_as_int(student_profile.get('total_msg_cnt', 0)),
        mainland_cnt=_as_int(student_profile.get('mainland_cnt', 0)),
        mainland_to_hk_cnt=_as_int(student_profile.get('mainland_to_hk_cnt', 0)),
        app_max_cnt=_as_int(student_profile.get('app_max_cnt', 0)),
        fraud_msisdn_present="YES - CONFIRMED" if str(student_profile.get('fraud_msisdn_present', '')).strip().lower() in ('true', '1', 'yes') else "No",
    )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,  # Lower for more focused output
            max_tokens=400,  # Reduced for concise scripts (~200 words)
            top_p=0.9,
            frequency_penalty=0.5,
        )
        content = response.choices[0].message.content.strip()
        return _trim_to_word_limit(content, max_words=200)  # Enforce 200 word limit
    except Exception as e:
        return f"⚠️ Error generating script: {str(e)}"
