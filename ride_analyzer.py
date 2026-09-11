"""
Ride/FTP Analyzer — prompt + API call
--------------------------------------
This script takes ALREADY-COMPUTED ride metrics (from fitparse or similar)
and asks an LLM to write a plain-English, coach-style summary.

Important design choice: the model is never asked to calculate anything.
It only interprets numbers you've already computed correctly. This avoids
the most common failure mode of AI sports-analysis tools — silently wrong math.
"""

import requests
import json
import time

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory and loads it into os.environ

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is not set.")
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable is not set.")

SYSTEM_PROMPT = """You are an experienced cycling coach reviewing a rider's
workout data. You write short, specific, encouraging but honest feedback —
the way a real coach would text a client after a ride, not like a generic
fitness app notification.

Rules:
- Never invent numbers. Only reference the metrics given to you.
- Be specific: reference actual numbers from the data, not vague praise.
- Keep it to 3-5 sentences plus one clear "focus for next time" suggestion.
- If the data suggests something notable (e.g. big HR drift, a new best
  effort, clear fatigue in the back half), say so plainly.
- Avoid generic filler like "great job" or "keep up the good work" unless
  the data specifically supports it.
- Write for an intermediate/advanced cyclist who understands terms like
  FTP, normalized power, and TSS — don't over-explain basics.
- If a segment breakdown is provided, use it to understand the shape of
  the ride (e.g. a climb in the middle, a fast finish). Never assume the
  whole ride matches one segment's character just because an aggregate
  stat (like total elevation gain) is high.
- If elapsed time and moving time differ significantly, base all analysis
  on moving time. Do not describe the ride as longer or more strenuous
  than its moving time actually supports.
- CRITICAL: If a metric (e.g. heart rate, power) is not present in the
  data provided, do NOT mention it, estimate it, or state any number for
  it — not even a plausible-sounding one. Only discuss metrics that
  literally appear in the data below. Inventing a number that isn't in
  the data is a serious error, worse than saying nothing about it.
- Respond in plain text only. Do not wrap your response in JSON, a code
  block, markdown formatting, or quotation marks. Write it exactly as
  you'd send a text message — plain sentences, nothing else.
"""

def build_user_prompt(metrics: dict) -> str:
    """
    metrics should be a dict you've already computed, e.g.:
    {
        "duration_min": 92,
        "ride_type": "endurance",  # optional, if known
        "avg_power_w": 187,
        "normalized_power_w": 204,
        "ftp_w": 250,
        "intensity_factor": 0.82,
        "tss": 103,
        "avg_hr": 148,
        "max_hr": 171,
        "hr_drift_pct": 6.2,        # HR drift from first half to second half
        "best_5s_w": 890,
        "best_1min_w": 410,
        "best_5min_w": 295,
        "best_20min_w": 232,
        "elevation_gain_m": 620,
    }
    """
    lines = []

    # Prefer moving time as the headline duration — it's what actually
    # reflects effort. Elapsed time is only mentioned if it differs
    # meaningfully, so the model doesn't confuse "on the bike" time with
    # "recording was running" time.
    moving_min = metrics.get("moving_time_min")
    elapsed_min = metrics.get("elapsed_time_min")
    if moving_min is not None:
        lines.append(f"Moving time: {moving_min} min")
        if elapsed_min is not None and (elapsed_min - moving_min) > 15:
            lines.append(
                f"Elapsed time: {elapsed_min} min (includes "
                f"{round(elapsed_min - moving_min, 1)} min stopped — "
                f"base your analysis on moving time, not elapsed time)"
            )
    elif elapsed_min is not None:
        lines.append(f"Duration: {elapsed_min} min")

    if metrics.get("ride_type"):
        lines.append(f"Ride type: {metrics['ride_type']}")

    # (field key, label, unit) — only included if the value is present.
    # This keeps GPX rides (no power data) from showing "None" everywhere;
    # the model only ever sees metrics that were actually computed.
    optional_fields = [
        ("avg_power_w", "Average power", "W"),
        ("normalized_power_w", "Normalized power", "W"),
        ("ftp_w", "FTP on file", "W"),
        ("intensity_factor", "Intensity Factor", ""),
        ("tss", "TSS", ""),
        ("avg_hr", "Average HR", "bpm"),
        ("max_hr", "Max HR", "bpm"),
        ("hr_drift_pct", "HR drift (1st half to 2nd half)", "%"),
        ("best_5s_w", "Best 5s power", "W"),
        ("best_1min_w", "Best 1min power", "W"),
        ("best_5min_w", "Best 5min power", "W"),
        ("best_20min_w", "Best 20min power", "W"),
        ("elevation_gain_m", "Elevation gain", "m"),
    ]
    for key, label, unit in optional_fields:
        value = metrics.get(key)
        if value is not None:
            lines.append(f"{label}: {value}{(' ' + unit) if unit else ''}")

    # Segment breakdown gives positional context — without this, the model
    # only sees one aggregate elevation_gain_m number and has no way to know
    # WHERE the climbing happened. A climb in the middle followed by a
    # descent looks identical, in aggregate stats, to a ride that's uphill
    # the whole way. This is what prevents that misread.
    segments = metrics.get("segments")
    if segments:
        lines.append("\nRide broken into equal time segments (use this to "
                      "understand the shape of the ride, e.g. where climbing "
                      "or hard efforts occurred — do not assume the whole "
                      "ride matches any single segment):")
        for seg in segments:
            parts = [f"Segment {seg['segment']} ({seg['elapsed_range_min'][0]}-{seg['elapsed_range_min'][1]} min)"]
            if seg.get("elevation_gain_m") is not None:
                parts.append(f"elevation gain {seg['elevation_gain_m']}m")
            if seg.get("avg_power_w") is not None:
                parts.append(f"avg power {seg['avg_power_w']}W")
            if seg.get("avg_hr") is not None:
                parts.append(f"avg HR {seg['avg_hr']}bpm")
            lines.append("  - " + ", ".join(parts))

    # Tell the model explicitly what's missing and why, so it doesn't
    # comment on the absence of power data as if something went wrong.
    if metrics.get("avg_power_w") is None:
        lines.append(
            "Note: No power meter data available for this ride (likely a "
            "GPS-only GPX file). Base the analysis on HR, pace/speed, and "
            "elevation instead — do not mention power or ask for a power meter."
        )

    return "Here is the ride data:\n\n" + "\n".join(lines) + \
        "\n\nWrite the coach summary now."


def analyze_ride(metrics: dict, groq_models: list = None, openrouter_models: list = None) -> str:
    """
    Tries Groq first (separate free-tier quota, much faster hardware, fixed
    30 req/min limit — no shared-pool congestion like OpenRouter free models
    have been hitting). Falls back to OpenRouter's multi-model chain only if
    every Groq model fails or GROQ_API_KEY isn't set.
    """
    if GROQ_API_KEY:
        try:
            return _call_groq(metrics, groq_models)
        except Exception as e:
            print(f"Groq failed, falling back to OpenRouter: {e}")
    else:
        print("No GROQ_API_KEY set — skipping Groq, using OpenRouter directly.")

    return _call_openrouter(metrics, openrouter_models)


def _call_groq(metrics: dict, models: list = None) -> str:
    """
    Groq has no built-in multi-model fallback like OpenRouter's `models`
    array, so we loop through the list ourselves and try each in order.
    Model IDs churn on Groq's catalog — verify current ones at
    console.groq.com/docs/models before relying on this long-term.
    """
    if models is None:
        models = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "llama-3.3-70b-versatile",
        ]

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for model in models:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(metrics)},
            ],
            "temperature": 0.7,
            "max_tokens": 1200,
        }
        # gpt-oss models on Groq are reasoning models — hide the reasoning
        # trace from the output so it doesn't eat the token budget or leak
        # into the response. Non-reasoning models ignore this harmlessly.
        if "gpt-oss" in model:
            payload["reasoning_format"] = "hidden"

        max_retries = 3
        for attempt in range(max_retries):
            resp = requests.post(GROQ_URL, headers=headers, data=json.dumps(payload))
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"Groq rate limited on {model} (attempt {attempt + 1}/{max_retries}). Waiting {wait}s...")
                time.sleep(wait)
                continue
            if not resp.ok:
                last_error = f"{model} -> {resp.status_code}: {resp.text}"
                print(f"Groq error: {last_error}")
                break  # try next model, don't keep retrying a hard failure
            result = resp.json()
            if "choices" not in result or not result["choices"]:
                last_error = f"{model} -> no choices in response: {result}"
                print(f"Groq error: {last_error}")
                break
            print(f"(answered by: groq/{model})")
            content = result["choices"][0]["message"]["content"]
            return _clean_response_text(content)
        else:
            last_error = f"{model} -> still rate-limited after retries"

    raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


def _call_openrouter(metrics: dict, models: list = None) -> str:
    """
    models: priority-ordered list of OpenRouter model IDs. OpenRouter tries
    them in order and automatically falls back to the next one on error
    (rate limits, downtime, content moderation, context-length issues).
    Put your most reliable/general-purpose model LAST as a safety net.
    """
    if models is None:
        models = [
            "nvidia/nemotron-3.5-lightning:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ]

    payload = {
        "models": models,  # note: plural "models", not "model"
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(metrics)},
        ],
        "temperature": 0.7,
        "max_tokens": 1200,
        "reasoning": {"exclude": True},
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    max_retries = 4
    for attempt in range(max_retries):
        resp = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload))
        if resp.status_code == 429:
            wait = 2 ** attempt  # 1, 2, 4, 8 seconds
            print(f"Rate limited (attempt {attempt + 1}/{max_retries}). Waiting {wait}s...")
            time.sleep(wait)
            continue
        if not resp.ok:
            print(f"Error {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        result = resp.json()

        if "choices" not in result or not result["choices"]:
            # Some failure modes return HTTP 200 with an error body instead
            # of a proper completion (e.g. all fallback models exhausted,
            # or a moderation/guardrail rejection). Surface what actually
            # came back instead of crashing on a bare KeyError.
            error_info = result.get("error", result)
            print(f"Unexpected response with no choices: {error_info}")
            raise RuntimeError(f"No usable response from any model in the fallback list: {error_info}")

        used_model = result.get("model", "unknown")
        print(f"(answered by: {used_model})")
        content = result["choices"][0]["message"]["content"]
        return _clean_response_text(content)
    raise RuntimeError("Still rate-limited after retries. Try again shortly, or switch models.")


def _clean_response_text(content: str) -> str:
    """
    Defensive cleanup for models that ignore the plain-text instruction.
    Some models wrap output as JSON (e.g. {"feedback": "..."}) despite
    being told not to. This unwraps that if detected, and also fixes
    literal '\\n' sequences that should be real line breaks.
    """
    text = content.strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                # Grab the first string value that looks like the actual
                # message (feedback/summary/response/text are common keys).
                for key in ("feedback", "summary", "response", "text", "message"):
                    if key in parsed and isinstance(parsed[key], str):
                        text = parsed[key]
                        break
                else:
                    # No known key matched — fall back to the first string value found.
                    string_values = [v for v in parsed.values() if isinstance(v, str)]
                    if string_values:
                        text = string_values[0]
        except json.JSONDecodeError:
            pass  # not actually JSON, leave as-is

    # Un-escape literal backslash-n sequences some models emit inside JSON strings
    text = text.replace("\\n\\n", "\n\n").replace("\\n", "\n")
    # Strip wrapping quotes if the whole thing got quoted like a literal string
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    return text.strip()


if __name__ == "__main__":
    import sys

    # Real-file mode: python ride_analyzer.py path/to/ride.fit [ftp]
    if len(sys.argv) >= 2:
        from ride_parser import load_ride, compute_metrics

        file_path = sys.argv[1]
        ftp = int(sys.argv[2]) if len(sys.argv) >= 3 else None

        records = load_ride(file_path)
        metrics = compute_metrics(records, ftp=ftp)
        for warning in metrics.pop("warnings", []):
            print(f"Note: {warning}")
        print(analyze_ride(metrics))
        sys.exit(0)

    # Sample-data mode (no args): for testing the prompt/model without a real file
    # Swap "model" below to compare outputs, e.g.
    #   "google/gemma-4-31b:free"  vs  "google/gemma-4-26b-a4b:free"
    sample_metrics = {
        "duration_min": 92,
        "ride_type": "endurance",
        "avg_power_w": 187,
        "normalized_power_w": 204,
        "ftp_w": 250,
        "intensity_factor": 0.82,
        "tss": 103,
        "avg_hr": 148,
        "max_hr": 171,
        "hr_drift_pct": 6.2,
        "best_5s_w": 890,
        "best_1min_w": 410,
        "best_5min_w": 295,
        "best_20min_w": 232,
        "elevation_gain_m": 620,
    }

    print(analyze_ride(sample_metrics))