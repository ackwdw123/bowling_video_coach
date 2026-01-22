import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import streamlit as st
import yaml

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class BowlerContext:
    bowler_name: str
    handedness: str            # "Right" or "Left"
    style: str                 # "1H" or "2H"
    experience_level: str      # "Beginner", "Intermediate", "Advanced", "Collegiate"
    primary_goal: str
    typical_miss: str
    leave: str                 # e.g., "10-pin", "flat 10", "4-pin", "bucket"
    lane_context: str          # e.g., "House shot, 41ft"
    camera_view: str           # "Side", "Back", "Front", "Other"
    notes: str


@dataclass
class FramePack:
    fps: float
    frame_indices: List[int]
    frames_bgr: List[np.ndarray]


# -----------------------------
# Utilities
# -----------------------------
def load_drills(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("drills", [])


def extract_frames(video_path: str, num_frames: int = 20) -> FramePack:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video (codec/format issue). Try converting to mp4 (H.264).")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total <= 0:
        # Fallback: read sequentially
        frames_tmp = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_tmp.append(frame)
        cap.release()
        total = len(frames_tmp)
        if total == 0:
            raise ValueError("Video appears empty or unreadable.")
        idxs = np.linspace(0, total - 1, min(num_frames, total)).astype(int).tolist()
        frames = [frames_tmp[i] for i in idxs]
        return FramePack(fps=fps, frame_indices=idxs, frames_bgr=frames)

    idxs = np.linspace(0, total - 1, min(num_frames, total)).astype(int).tolist()
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
        else:
            # placeholder if a frame can't be read
            frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))

    cap.release()
    return FramePack(fps=fps, frame_indices=idxs, frames_bgr=frames)


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# -----------------------------
# Coach logic (rule-based v1)
# -----------------------------
def infer_tags(ctx: BowlerContext) -> List[str]:
    """
    Convert user-provided symptoms into normalized tags.
    This is v1 and intentionally simple; later you can incorporate pose metrics.
    """
    tags = []

    miss = (ctx.typical_miss or "").lower()
    leave = (ctx.leave or "").lower()
    view = (ctx.camera_view or "").lower()

    # Miss direction
    if "right" in miss or "miss right" in miss:
        tags.append("miss_right")
    if "left" in miss or "miss left" in miss:
        tags.append("miss_left")
    if "high" in miss:
        tags.append("high_hit")
    if "light" in miss:
        tags.append("light_hit")

    # Leaves / ball reaction cues
    if "10" in leave:
        # could be flat 10 or corner 10 - both often match release/entry angle issues
        tags.append("leave_10")
        if "flat" in leave:
            tags.append("flat_10")
    if "7" in leave:
        tags.append("leave_7")
    if "bucket" in leave:
        tags.append("bucket_leave")

    # Common coaching buckets based on typical complaints
    if "inconsistent" in miss or "all over" in miss or "spray" in miss:
        tags.append("targeting")
        tags.append("tempo")
    if "pulled" in miss or "pull" in miss:
        tags.append("pulling")
        tags.append("opening_shoulders")
    if "timing" in miss or "late" in miss:
        tags.append("late_timing")
    if "early" in miss:
        tags.append("early_timing")

    # If they mention balance in notes
    notes = (ctx.notes or "").lower()
    if "balance" in notes or "falling" in notes or "fall off" in notes:
        tags.append("balance")
        tags.append("posting")

    # If no strong signal, default to fundamentals
    if not tags:
        tags = ["targeting", "tempo", "balance"]

    # Camera hint (if not side, suggest what to film next)
    if "side" not in view:
        tags.append("need_side_view")

    # style-specific nuance
    if ctx.style == "2H":
        tags.append("two_handed")
    else:
        tags.append("one_handed")

    # de-dupe
    return sorted(set(tags))


def select_drills(drills: List[Dict[str, Any]], tags: List[str], where: str) -> List[Dict[str, Any]]:
    """
    Select drills matching tags and location constraint (home/lanes).
    """
    scored = []
    for d in drills:
        d_tags = set(d.get("tags", []))
        score = len(d_tags.intersection(tags))
        if score <= 0:
            continue
        if d.get("where") != where:
            continue
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:3]]  # top 3 per location


def build_priorities(tags: List[str], ctx: BowlerContext) -> List[Dict[str, Any]]:
    """
    Translate tags into a coach-style prioritized list.
    """
    priorities = []

    # Timing
    if "late_timing" in tags:
        priorities.append({
            "issue": "Timing appears late (ball arrives after the slide / body gets ahead of the ball).",
            "why": "Late timing often causes pulling the ball, inconsistent release, and weaker continuation because the body has to ‘wait’ or ‘save’ the shot.",
            "implement": [
                "Start the pushaway with step 1 (or slightly earlier) and let gravity start the swing.",
                "Keep steps 1–3 smooth; avoid rushing the last two steps.",
                "Feel: ‘ball falls, feet match it’ — not the other way around."
            ],
            "cue": "“Start earlier. Let it fall.”",
            "confirm_next": "Film a true side view (waist height) showing feet + ball the entire approach."
        })

    if "early_timing" in tags:
        priorities.append({
            "issue": "Timing appears early (ball gets to the bottom too soon / you catch up to it).",
            "why": "Early timing can force muscling at the bottom, cause lofting, and reduce repeatability when you try to ‘hold’ the ball.",
            "implement": [
                "Delay the pushaway slightly and keep the first step calm.",
                "Focus on a longer, smoother swing cadence rather than ‘placing’ the ball.",
                "Make sure the slide is committed so the ball can pass the ankle naturally."
            ],
            "cue": "“Smooth first step.”",
            "confirm_next": "Film side view; look for ball at top of backswing occurring near step 3–4 (5-step approach)."
        })

    # Directional misses / targeting
    if "miss_right" in tags:
        priorities.append({
            "issue": "Miss trend: right (often from projection, timing, or early opening).",
            "why": "Right misses usually come from getting around it early, dropping the shoulder, or not projecting through the target line.",
            "implement": [
                "Keep your head and sternum stable; project through your target, not at the pins.",
                "Feel the hand stay ‘behind/inside’ the ball longer through release (2H: stable wrist/forearm line).",
                "Hold the finish for a 2-count to prevent bailing out."
            ],
            "cue": "“Project through the target.”",
            "confirm_next": "Film from behind (centered) to see swing direction and drift."
        })

    if "miss_left" in tags or "pulling" in tags:
        priorities.append({
            "issue": "Miss trend: pulled / left (often from early shoulders or grabbing at release).",
            "why": "Pulled shots commonly come from opening shoulders early, muscling down, or steering at the bottom.",
            "implement": [
                "Keep shoulders ‘closed’ a fraction longer—let the swing pass the ankle before rotating.",
                "Feel the thumb/hand exit clean and then follow through toward your target.",
                "Relax the downswing: no ‘hit’ at the bottom."
            ],
            "cue": "“Swing past the ankle, then rotate.”",
            "confirm_next": "Film behind view + side view to see shoulder timing."
        })

    # Balance/posting
    if "balance" in tags or "posting" in tags:
        priorities.append({
            "issue": "Finish/posting needs strengthening (stability at the line).",
            "why": "If you can’t post the shot, you’ll compensate with the arm—accuracy and repeatability drop fast.",
            "implement": [
                "Slow the approach to 60–70% and freeze the finish for a full 2-count.",
                "Stack: head over shoulder over knee; trail leg behind for counterbalance.",
                "Eyes stay on target until the ball hits the arrows."
            ],
            "cue": "“Hold the finish.”",
            "confirm_next": "Film side view; check if head stays level through release."
        })

    # If nothing got added (rare), add a fundamentals priority
    if not priorities:
        priorities.append({
            "issue": "Fundamentals focus: repeatable tempo + targeting.",
            "why": "If tempo and targeting aren’t stable, all other changes become hard to measure.",
            "implement": [
                "Use a consistent pre-shot routine and commit to one target.",
                "Practice 3-6-9 targeting moves to build adjustment confidence.",
                "Film from the side to validate timing and posture."
            ],
            "cue": "“Same tempo, same target.”",
            "confirm_next": "Film side view and behind view for a complete baseline."
        })

    # Keep it “Gold coach style”: max 3 priorities
    return priorities[:3]


def generate_report(ctx: BowlerContext, drills: List[Dict[str, Any]]) -> Dict[str, Any]:
    tags = infer_tags(ctx)
    priorities = build_priorities(tags, ctx)

    home_drills = select_drills(drills, tags, where="home")
    lane_drills = select_drills(drills, tags, where="lanes")

    quick_wins = [
        "Pick ONE cue per session (don’t stack cues).",
        "Film 3 shots from the same angle every session to track progress.",
        "Hold your finish for a 2-count on all practice shots."
    ]

    if "need_side_view" in tags:
        quick_wins.insert(0, "If possible, re-film from the SIDE (waist height) showing feet + ball the whole approach.")

    next_session_plan = [
        "Home (5–10 min): No-step swings or metronome footwork (easy reps, clean tempo).",
        "Lanes (15–20 min): One-step release drill + posting (hold finish).",
        "Lanes (15 min): 3-6-9 targeting (measure repeatability, not score)."
    ]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "tags": tags,
        "context": asdict(ctx),
        "priorities": priorities,
        "home_drills": home_drills,
        "lane_drills": lane_drills,
        "quick_wins": quick_wins,
        "next_session_plan": next_session_plan
    }


# -----------------------------
# PDF report
# -----------------------------
def wrap_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, line_height: float) -> float:
    """
    Draw wrapped text and return the new y position.
    """
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = w
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y


def make_pdf(report: Dict[str, Any], output_path: str) -> None:
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    margin = 0.75 * inch
    x = margin
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Bowling Video Coach Report")
    y -= 0.3 * inch

    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated: {report['generated_at']}")
    y -= 0.25 * inch

    ctx = report["context"]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Bowler Context")
    y -= 0.18 * inch
    c.setFont("Helvetica", 10)

    ctx_lines = [
        f"Name: {ctx.get('bowler_name','')}",
        f"Handedness: {ctx.get('handedness','')} | Style: {ctx.get('style','')} | Level: {ctx.get('experience_level','')}",
        f"Primary goal: {ctx.get('primary_goal','')}",
        f"Typical miss: {ctx.get('typical_miss','')}",
        f"Common leave: {ctx.get('leave','')}",
        f"Lane context: {ctx.get('lane_context','')}",
        f"Camera view: {ctx.get('camera_view','')}",
    ]
    for line in ctx_lines:
        y = wrap_text(c, line, x, y, width - 2 * margin, 12)

    if ctx.get("notes"):
        y -= 6
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Notes:")
        y -= 12
        c.setFont("Helvetica", 10)
        y = wrap_text(c, ctx["notes"], x, y, width - 2 * margin, 12)

    y -= 0.15 * inch

    # Priorities
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Coach Priorities (Top 3)")
    y -= 0.2 * inch

    for i, p in enumerate(report["priorities"], start=1):
        if y < margin + 2 * inch:
            c.showPage()
            y = height - margin

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, f"{i}) {p['issue']}")
        y -= 0.18 * inch

        c.setFont("Helvetica", 10)
        y = wrap_text(c, f"Why it matters: {p['why']}", x, y, width - 2 * margin, 12)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "How to implement:")
        y -= 12
        c.setFont("Helvetica", 10)
        for step in p["implement"]:
            y = wrap_text(c, f"- {step}", x + 12, y, width - 2 * margin - 12, 12)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, f"Cue: {p['cue']}")
        y -= 14
        c.setFont("Helvetica", 10)
        y = wrap_text(c, f"Film next: {p['confirm_next']}", x, y, width - 2 * margin, 12)
        y -= 0.12 * inch

    # Drills
    def render_drill_block(title: str, drill_list: List[Dict[str, Any]], y: float) -> float:
        nonlocal c
        if y < margin + 2 * inch:
            c.showPage()
            y = height - margin

        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 0.2 * inch

        if not drill_list:
            c.setFont("Helvetica", 10)
            c.drawString(x, y, "No drills matched your inputs. Use fundamentals: one-step + posting + targeting.")
            y -= 0.2 * inch
            return y

        for d in drill_list:
            if y < margin + 1.8 * inch:
                c.showPage()
                y = height - margin

            c.setFont("Helvetica-Bold", 11)
            c.drawString(x, y, d["name"])
            y -= 0.16 * inch

            c.setFont("Helvetica", 10)
            y = wrap_text(c, f"Reps: {d.get('reps','')}", x, y, width - 2 * margin, 12)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, y, "Steps:")
            y -= 12
            c.setFont("Helvetica", 10)
            for s in d.get("steps", []):
                y = wrap_text(c, f"- {s}", x + 12, y, width - 2 * margin - 12, 12)

            y = wrap_text(c, f"Success: {d.get('success','')}", x, y, width - 2 * margin, 12)
            y -= 0.12 * inch

        return y

    y = render_drill_block("Home Drills (Solo)", report["home_drills"], y)
    y = render_drill_block("Lane Drills (Solo)", report["lane_drills"], y)

    # Quick wins + plan
    if y < margin + 2 * inch:
        c.showPage()
        y = height - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Quick Wins")
    y -= 0.2 * inch
    c.setFont("Helvetica", 10)
    for q in report["quick_wins"]:
        y = wrap_text(c, f"- {q}", x, y, width - 2 * margin, 12)

    y -= 0.1 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Next Session Plan")
    y -= 0.2 * inch
    c.setFont("Helvetica", 10)
    for s in report["next_session_plan"]:
        y = wrap_text(c, f"- {s}", x, y, width - 2 * margin, 12)

    c.save()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Bowling Video Coach", layout="wide")
st.title("🎳 Bowling Video Coach (Local v1)")

# Load drill library
DRILLS_PATH = "drills.yaml"
if not os.path.exists(DRILLS_PATH):
    st.error("Missing drills.yaml in the same folder as app.py")
    st.stop()

drills = load_drills(DRILLS_PATH)

with st.sidebar:
    st.header("Bowler context")
    bowler_name = st.text_input("Bowler name", "")
    handedness = st.selectbox("Handedness", ["Right", "Left"])
    style = st.selectbox("Style", ["1H", "2H"])
    experience_level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced", "Collegiate"])
    primary_goal = st.text_input("Primary goal", "Stop missing right / improve spare conversion")
    typical_miss = st.text_input("Typical miss description", "Misses 10-pin right by a few inches")
    leave = st.text_input("Common leave (if any)", "10-pin")
    lane_context = st.text_input("Lane context", "House shot (unknown length is fine)")
    camera_view = st.selectbox("Camera view", ["Side", "Back", "Front", "Other"])
    notes = st.text_area("Notes (optional)", "")

uploaded = st.file_uploader("Upload a bowler video (mp4/mov/m4v)", type=["mp4", "mov", "m4v"])

colA, colB = st.columns([1, 1])

if uploaded:
    ctx = BowlerContext(
        bowler_name=bowler_name.strip(),
        handedness=handedness,
        style=style,
        experience_level=experience_level,
        primary_goal=primary_goal.strip(),
        typical_miss=typical_miss.strip(),
        leave=leave.strip(),
        lane_context=lane_context.strip(),
        camera_view=camera_view,
        notes=notes.strip()
    )

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    with colA:
        st.subheader("Video")
        st.video(video_path)

    with colB:
        st.subheader("Frame sampling")
        num_frames = st.slider("Frames to sample for preview", 8, 40, 20)
        show_frames = st.checkbox("Show sampled frames", value=True)

    if st.button("Analyze + Generate Report"):
        try:
            with st.spinner("Extracting frames..."):
                framepack = extract_frames(video_path, num_frames=num_frames)

            if show_frames:
                st.subheader("Sampled frames (for quick visual reference)")
                cols = st.columns(4)
                for i, (idx, frame) in enumerate(zip(framepack.frame_indices, framepack.frames_bgr)):
                    with cols[i % 4]:
                        st.image(bgr_to_rgb(frame), caption=f"Frame {idx}", use_container_width=True)

            with st.spinner("Generating coach report..."):
                report = generate_report(ctx, drills)

            st.success("Report generated.")

            # Display report
            st.subheader("Coach Priorities")
            for i, p in enumerate(report["priorities"], start=1):
                st.markdown(f"### {i}) {p['issue']}")
                st.markdown(f"**Why it matters:** {p['why']}")
                st.markdown(f"**Cue:** {p['cue']}")
                st.markdown("**How to implement:**")
                for step in p["implement"]:
                    st.write(f"- {step}")
                st.markdown(f"**Film next:** {p['confirm_next']}")

            st.subheader("Drills (Home)")
            if report["home_drills"]:
                for d in report["home_drills"]:
                    st.markdown(f"**{d['name']}** — {d.get('reps','')}")
                    for s in d.get("steps", []):
                        st.write(f"- {s}")
                    st.write(f"Success: {d.get('success','')}")
            else:
                st.write("No matched home drills — use fundamentals (no-step swings + metronome).")

            st.subheader("Drills (Lanes)")
            if report["lane_drills"]:
                for d in report["lane_drills"]:
                    st.markdown(f"**{d['name']}** — {d.get('reps','')}")
                    for s in d.get("steps", []):
                        st.write(f"- {s}")
                    st.write(f"Success: {d.get('success','')}")
            else:
                st.write("No matched lane drills — use one-step + posting + targeting.")

            st.subheader("Quick Wins")
            for q in report["quick_wins"]:
                st.write(f"- {q}")

            st.subheader("Next Session Plan")
            for s in report["next_session_plan"]:
                st.write(f"- {s}")

            # Make PDF
            with st.spinner("Building PDF..."):
                pdf_name = f"bowling_coach_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_path = os.path.join(tempfile.gettempdir(), pdf_name)
                make_pdf(report, pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=f,
                    file_name=pdf_name,
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            # Best-effort cleanup
            try:
                os.remove(video_path)
            except Exception:
                pass

else:
    st.info("Upload a video to begin. Best filming: SIDE view at waist height, showing feet + ball the entire approach.")

