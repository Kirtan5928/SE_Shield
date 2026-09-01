"""
layer3_slm/config_layer3.py
============================
Central configuration for the Layer 3 NLI engine.

Design notes
------------
* HYPOTHESIS_TEMPLATES are the core decision boundary of the NLI approach.
  Each template is the natural-language description that the model will score
  for entailment against the incoming message.  Changing a template changes
  the model's classification behaviour — edit with care and re-run evaluate_nli.py
  after any change.

* ATTACK_LABELS is the live label set.  To add a new attack type:
    1. Add the label string to ATTACK_LABELS.
    2. Add a corresponding entry to HYPOTHESIS_TEMPLATES.
    3. Optionally extend the signal word lists if the attack type has
       characteristic vocabulary.
    No retraining, no data labelling required.

* Signal word lists are used ONLY by the ExplanationEngine to generate the
  `reason` field.  They do NOT affect classification — that is NLI-only.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
MAX_LENGTH  = 256          # tokeniser truncation; matches Layer 2 preprocessing

# LAYER2_THRESHOLD = 0 means NLI runs on every message Layer 2 passes through.
# Only messages where LR gives literally 0% attack probability bypass NLI
# (int(0.000 × 100) = 0) — which is statistically near-impossible.
#
# WHY 0 AND NOT 50:
# In L2-gated evaluation, 40/324 attacks had risk_score ≤ 50 and were never
# seen by Layer 3. Layer 2's recall ceiling became Layer 3's FN.
# Setting threshold=0 delegates the benign/attack decision entirely to NLI
# for every message Layer 2 touches, eliminating this failure mode.
# Force-NLI mode already proved FN=0 — this makes that the production default.
#
# COST: ~150ms NLI inference on every message vs fast-path for low-risk ones.
# ACCEPTABLE because: this is a conversation-level pipeline, not a bulk scanner.
# Layer 2 SVM still eliminates obvious benign traffic before Layer 3 is invoked.
LAYER2_THRESHOLD = 0

# When the top NLI confidence across all labels is below this value the
# sub-type prediction is too uncertain to trust.  The pipeline falls back
# to the generic label "phishing" (safest conservative call).
MIN_SUBTYPE_CONFIDENCE = 0.25

# Signal-gated benign recovery threshold.
#
# Signal-gated benign recovery — DISABLED (threshold = 0.0)
#
# Originally designed for: benign messages Layer 2 over-flagged at risk_score
# 51–70, which reached NLI and scored mildly suspicious (conf 0.25–0.49)
# despite having no SE signals. Overriding those to benign improved precision.
#
# WHY DISABLED: With LAYER2_THRESHOLD=0, Layer 3 now processes every message,
# including subtle attacks with no SE vocabulary. These score identically to
# the benign false-positives the recovery was designed for: no signals + NLI
# confidence 0.25–0.49. The recovery cannot distinguish them, so it was
# catching 229 actual attacks and calling them benign (recall dropped to 0.24).
#
# Setting 0.0 means `clf["confidence"] < 0.0` is never true — recovery
# never fires. NLI output is used as-is for all messages.
#
# ACCEPTED TRADEOFF: Precision ~0.65 (some benign messages get mild attack
# labels from NLI). Recall = 1.0000. For a security dashboard, FN=0 is the
# hard requirement — false alarms are recoverable, missed attacks are not.
SIGNAL_GATED_BENIGN_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Top-labels config  (multi-label output — additive, does not replace `label`)
#
# SE attacks are taxonomically overlapping.  A message entailing both
# "credential_harvesting" and "phishing" is correctly described by both.
# `top_labels` surfaces up to TOP_LABELS_MAX_COUNT attack types that score
# above meaningful thresholds.  The primary `label` field is unchanged.
#
# Inclusion rules (both must pass):
#   1. confidence >= TOP_LABELS_MIN_CONF           (absolute floor)
#   2. confidence >= top_conf * TOP_LABELS_REL_FLOOR  (relative floor)
#
# Rule 2 prevents noise labels appearing when one label dominates
# (e.g. if top=0.99, second needs ≥ 0.35 to appear — not just 0.15).
# When the model is uncertain (top=0.30), rule 2 allows any label ≥ 0.10.
# ---------------------------------------------------------------------------
TOP_LABELS_MAX_COUNT  = 3      # maximum labels in top_labels list
TOP_LABELS_MIN_CONF   = 0.15   # absolute minimum confidence to appear
TOP_LABELS_REL_FLOOR  = 0.35   # must be ≥ this fraction of the top label's score

# ---------------------------------------------------------------------------
# Label taxonomy  (extend freely — no retraining required)
#
# ARCHITECTURAL NOTE — WHY "benign" IS NOT IN THIS LIST:
#   "benign" is not an attack sub-type. Layer 3's sole job is to classify
#   WHICH TYPE of attack a suspicious message is. The benign/attack binary
#   decision belongs to Layer 2 (SVM), not Layer 3.
#
#   Including "benign" as an NLI label caused catastrophic FN because
#   sophisticated BEC and pretexting attacks mimic professional language
#   ("following up on our meeting, please process the invoice") which
#   directly entails any benign hypothesis describing business communication.
#   The NLI model cannot distinguish "benign follow-up" from "BEC attack
#   disguised as a follow-up" via hypothesis entailment alone.
#
#   The benign passthrough in layer3_pipeline.py still exists — it fires
#   when Layer 2's SVM classifies the message as "benign" (not attack).
#   That is the correct, trained binary decision for benign detection.
# ---------------------------------------------------------------------------

ATTACK_LABELS: list[str] = [
    "phishing",
    "spear_phishing",
    "pretexting",
    "credential_harvesting",
    "baiting",
    "vishing",
    "business_email_compromise",
]

# ---------------------------------------------------------------------------
# Hypothesis templates
#
# Crafting principles:
#   1. DISCRIMINATIVE over descriptive. Each hypothesis must name what is
#      UNIQUE to this label — not the traits it shares with other SE types.
#      "Uses urgency" is shared by phishing, BEC, vishing. "Impersonates a
#      bank via phone/SMS" is unique to vishing.
#   2. CONCRETE over abstract. Name the specific mechanism: "directs the
#      recipient to a fake login page" beats "attempts to steal credentials."
#   3. BENIGN must be POSITIVELY framed. "This is a routine, legitimate
#      message" outperforms "no deceptive intent" because the model anchors
#      on what the text IS, not what it lacks.
#   4. SPEAR_PHISHING requires explicit personalisation markers. Do NOT use
#      vague phrases like "contextual details" — these match any email.
#      Require named individual + specific role/project/relationship.
#   5. Keep each hypothesis under ~55 tokens. Longer hypotheses dilute the
#      entailment signal and tend to match more labels simultaneously.
#
# Tuning procedure:
#   After changing any template, run:
#       python layer3_slm/src/evaluate_nli.py --split val --n 500
#   and compare recall_attack + sub_type_distribution.
#   Use run_layer3.py --smoke-test for quick sanity checks.
# ---------------------------------------------------------------------------

HYPOTHESIS_TEMPLATES: dict[str, str] = {
    # ── Design rules ─────────────────────────────────────────────────────────
    # 1. ≤ 25 words per hypothesis. Longer hypotheses cause cross-label bleed.
    # 2. Single concrete mechanism per hypothesis. No "or" clauses.
    # 3. Unique vocabulary across hypotheses. Overlapping words = overlapping scores.
    # NOTE: "benign" has been removed. Layer 2 SVM handles benign/attack gating.
    #       NLI only classifies which ATTACK SUB-TYPE the message is.
    # ─────────────────────────────────────────────────────────────────────────

    # Unique anchors: "account will be suspended", "click a link", "verify identity"
    # Phishing is the GENERIC attack label — covers mass impersonation with urgency
    # that doesn't fit a more specific category. Serves as the correct residual.
    "phishing": (
        "The sender impersonates a service or authority figure and creates "
        "urgency — threatening account suspension, security breaches, or "
        "legal consequences — to pressure the recipient into clicking a link "
        "or taking immediate action."
    ),

    # Unique anchors: "researched", "recipient's actual name", "named manager",
    # "specific internal project".
    # INTENTIONALLY NARROW: spear_phishing must require explicit proof that the
    # attacker researched this specific individual — a named person, their manager,
    # their team, or a specific deal they are known to be working on.
    # Generic business language, urgency, or vague "personal context" must NOT
    # qualify. This prevents spear_phishing from acting as a catch-all residual
    # label when other hypotheses don't strongly fire.
    "spear_phishing": (
        "The attacker has researched this specific individual beforehand and "
        "proves it by mentioning the recipient's actual name, their manager, "
        "a named colleague, or a specific internal project or deal they are "
        "currently working on — using this insider knowledge to make the "
        "request appear legitimate."
    ),

    # Unique anchors: "IT support", "login credentials", "remote access"
    # Fabricated technical identity + request for access is the sole mechanism.
    "pretexting": (
    "The sender is impersonating a trusted authority figure — "
    "such as a bank official, government representative, law "
    "enforcement, or IT personnel — using a fabricated scenario "
    "such as account suspension, fraud investigation, or security "
    "audit to manipulate the recipient into compliance."
    ),

    # Unique anchors: "share", "password", "username", "one-time code"
    # Mechanism = the ASK itself. No impersonation requirement — peer,
    # colleague, stranger, or authority: requesting credentials is the attack.
    "credential_harvesting": (
        "The sender asks the recipient to share or send their password, "
        "username, login details, account access, PIN, or one-time code."
    ),

    # Unique anchors: "won a prize", "gift card", "claim"
    # NO mention of links or downloads — those cause false positives.
    # The lure (prize/reward) is the sole discriminating mechanism.
    "baiting": (
        "The sender tells the recipient they have won a prize or gift card "
        "and asks them to claim their free reward."
    ),

    # Unique anchors: "bank", "call", "phone number", "security alert"
    # Phone/SMS channel is the key discriminator vs. email-based attacks.
    "vishing": (
        "The sender impersonates a bank and claims there is a security alert "
        "that requires the recipient to call a specific phone number."
    ),

    # Unique anchors: "wire transfer", "executive", "payment account"
    # No mention of links — BEC works via authority, not deceptive URLs.
    "business_email_compromise": (
        "The sender claims to be a company executive and requests an urgent "
        "wire transfer or a change to the company payment account details."
    ),
}

# ---------------------------------------------------------------------------
# SE signal word lists  (used ONLY by ExplanationEngine for `reason` generation)
# ---------------------------------------------------------------------------

URGENCY_SIGNALS: list[str] = [
    r"\burgent\b",
    r"\bimmediately\b",
    r"\basap\b",
    r"\bact now\b",
    r"\bdeadline\b",
    r"\bexpir",          # expire / expiring / expiration
    r"\bsuspend",        # suspend / suspended / suspension
    r"\blimited time\b",
    r"\btoday only\b",
    r"\blast chance\b",
    r"\bfinal notice\b",
    r"\boverdue\b",
    r"\bwithin \d+ hour",
    r"\bwithin \d+ minute",
    r"\btime.sensitive\b",
]

CREDENTIAL_SIGNALS: list[str] = [
    r"\bpassword\b",
    r"\busername\b",
    r"\blog.?in\b",
    r"\bsign.?in\b",
    r"\bcredentials?\b",
    r"\bverify your\b",
    r"\bconfirm your\b",
    r"\bauthenticat",
    r"\btwo.factor\b",
    r"\b2fa\b",
    r"\botp\b",
    r"\bsecurity code\b",
    r"\breset your (password|account)\b",
    r"\bupdate your (account|password|details|information)\b",
]

AUTHORITY_SIGNALS: list[str] = [
    r"\bceo\b",
    r"\bcfo\b",
    r"\bdirector\b",
    r"\bpresident\b",
    r"\bmanager\b",
    r"\bit (department|support|team|helpdesk|admin)\b",
    r"\bhuman resources\b",
    r"\bhr department\b",
    r"\birs\b",
    r"\bfederal\b",
    r"\bgovernment\b",
    r"\bbank\b",
    r"\bcompliance\b",
    r"\blegal (team|department)\b",
    r"\badministrator\b",
    r"\bsupport team\b",
    r"\bsecurity team\b",
]

FINANCIAL_SIGNALS: list[str] = [
    r"\bwire transfer\b",
    r"\btransfer\b",
    r"\bpayment\b",
    r"\binvoice\b",
    r"\bfunds\b",
    r"\brouting number\b",
    r"\baccount number\b",
    r"\bpayroll\b",
    r"\brefund\b",
    r"\btax\b",
    r"\bpurchase order\b",
    r"\bpo number\b",
    r"\bremittance\b",
]

THREAT_SIGNALS: list[str] = [
    r"\bsuspended\b",
    r"\bblocked\b",
    r"\bdisabled\b",
    r"\bterminated\b",
    r"\bviolation\b",
    r"\billegal\b",
    r"\bunauthori[sz]ed\b",
    r"\bbreach\b",
    r"\bcompromised\b",
    r"\blocked out\b",
    r"\baccess (denied|revoked)\b",
    r"\bcriminal\b",
    r"\barrest\b",
]

LINK_ACTION_SIGNALS: list[str] = [
    r"\bclick here\b",
    r"\bclick the link\b",
    r"\bfollow (this|the) link\b",
    r"\bdownload\b",
    r"\bopen (the )?attachment\b",
    r"\bverify your account\b",
    r"\bconfirm your email\b",
    r"https?://\S+",
]