#!/usr/bin/env python3
"""MAPoRL Episode Simulation Dashboard"""
import gradio as gr

# ── Color palette (light theme) ───────────────────────────────────────────────
BG      = "#f8f9ff"
CARD    = "#ffffff"
TEXT    = "#1e1e2e"
SUB     = "#4a4a6a"
BORDER  = "#dde1f0"
BLUE    = "#3b82f6"
BLUE_BG = "#eff6ff"
GREEN   = "#16a34a"
GREEN_BG= "#f0fdf4"
RED     = "#dc2626"
AMBER   = "#d97706"
AMBER_BG= "#fffbeb"
PURPLE  = "#7c3aed"
PURPLE_BG="#f5f3ff"
TEAL    = "#0891b2"
TEAL_BG = "#ecfeff"

# ── Episode data ──────────────────────────────────────────────────────────────
QUESTION = ("Janet's ducks lay 16 eggs per day. "
            "She eats 3 for breakfast every morning and bakes 4 into muffins. "
            "She sells the remainder at $2 each at the farmers' market. "
            f"<b style='color:{TEXT}'>How much does she make every day?</b>")

T0_A0 = (f"<span style='color:{SUB}'>Let me calculate egg consumption first:</span><br>"
          f"• Breakfast: 3 &nbsp;• Muffins: 4<br>"
          f"Eggs for sale: 16 − 3 − 4 = <b>9</b><br>"
          f"Revenue: 9 × $2 = <b style='color:{GREEN}'>$18 ✓</b>")

T0_A1 = (f"<span style='color:{SUB}'>Janet eats 3 for breakfast, so</span><br>"
          f"16 − 3 = 13 eggs remain.<br>"
          f"13 × $2 = <b style='color:{RED}'>$26</b>")

T1_A0 = (f"Agent 1 proposed <b>$26</b>,<br>"
          f"but it seems the 4 muffin eggs were not subtracted.<br><br>"
          f"16 − 3 (breakfast) − 4 (muffins) = 9 eggs<br>"
          f"9 × $2 = <b style='color:{GREEN}'>$18 ← correct answer.</b>")

T1_A1 = (f"Looking at Agent 0's solution, I made a mistake<br>"
          f"by not subtracting the 4 muffin eggs. Correcting:<br><br>"
          f"16 − 3 − 4 = 9 eggs → 9 × $2 = <b style='color:{GREEN}'>$18</b><br>"
          f"Agent 0 is right. Answer: <b style='color:{GREEN}'>$18 ✓</b>")

# ── HTML builders ─────────────────────────────────────────────────────────────
NAV = ["Problem", "Turn 0 — A0", "Turn 0 — A1", "Verifier (T0)",
       "Turn 1 — A0", "Turn 1 — A1", "Verifier (T1)", "Bonus Rule", "Final Reward", "PPO Update"]

def S(css): return f"style='{css}'"

def wrap(step, content):
    pct   = int(step / 9 * 100)
    label = NAV[step]
    progress = (
        f"<div {S(f'color:{SUB};font-size:11px;margin-bottom:4px')}>"
        f"Step {step+1} / 10 — {label}</div>"
        f"<div {S(f'background:{BORDER};border-radius:6px;height:5px;margin-bottom:12px')}>"
        f"<div {S(f'background:{BLUE};border-radius:6px;height:5px;width:{pct}%')}></div></div>"
    )
    return (
        f"<div {S(f'font-family:Segoe UI,Arial,sans-serif;max-width:860px;margin:0 auto')}>"
        f"<div {S(f'background:{BG};border:1px solid {BORDER};border-radius:14px;padding:22px 24px;min-height:350px')}>"
        f"{progress}{content}"
        f"</div></div>"
    )

def title(text, color):
    return f"<div {S(f'font-size:15px;font-weight:700;color:{color};margin-bottom:2px')}>{text}</div>"

def q_full():
    return (
        f"<div {S(f'background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:16px 18px')}>"
        f"<div {S(f'color:{TEAL};font-size:10px;font-weight:700;letter-spacing:1px;margin-bottom:8px')}>📋 QUESTION (GSM8K)</div>"
        f"<div {S(f'color:{TEXT};font-size:14px;line-height:1.75')}>{QUESTION}</div>"
        f"<div {S(f'color:{SUB};font-size:12px;margin-top:10px')}>Answer: (16−3−4)×$2 = <b style='color:{GREEN}'>$18</b></div>"
        f"</div>"
    )

def q_mini():
    return (
        f"<div {S(f'background:{CARD};border:1px solid {BORDER};border-radius:7px;padding:9px 14px;color:{SUB};font-size:12.5px')}>"
        f"<b style='color:{TEXT}'>Q:</b> {QUESTION}</div>"
    )

def agent(aid, text, score=None, ok=None, bonus=None, faded=False):
    bc  = BLUE if aid == 0 else GREEN
    bgc = BLUE_BG if aid == 0 else GREEN_BG
    tag = f"🤖 AGENT {aid}"
    op  = "0.35" if faded else "1"
    sc_html = ""
    if score is not None:
        sc_c  = GREEN if ok else RED
        sc_bg = "#dcfce7" if ok else "#fee2e2"
        sym   = "✓" if ok else "✗"
        sc_html = (f"<span {S(f'display:inline-block;background:{sc_bg};color:{sc_c};'
                              f'border-radius:4px;padding:2px 9px;font-size:11.5px;font-weight:700;margin-top:9px')}>"
                   f"Verifier: {score:.2f} {sym}</span>")
    bonus_html = ""
    if bonus is not None:
        bonus_html = (f"<span {S(f'display:inline-block;background:#ede9fe;color:{PURPLE};'
                                 f'border-radius:4px;padding:2px 9px;font-size:11.5px;font-weight:700;margin-top:9px;margin-left:5px')}>"
                      f"+α₁ bonus: +{bonus}</span>")
    text_html = f"<div {S(f'color:{TEXT};font-size:13.5px;line-height:1.65')}>{text}</div>"
    return (
        f"<div {S(f'flex:1;background:{bgc};border-left:3px solid {bc};'
                  f'border-radius:9px;padding:13px 15px;opacity:{op}')}>"
        f"<div {S(f'color:{bc};font-size:10px;font-weight:700;letter-spacing:1px;margin-bottom:7px')}>{tag}</div>"
        f"{text_html}"
        f"{sc_html}{bonus_html}</div>"
    )

def row(*cols):
    inner = "".join(cols)
    return f"<div {S('display:flex;gap:12px')}>{inner}</div>"

def card(label, lcolor, bgc, bdc, body):
    return (
        f"<div {S(f'background:{bgc};border:1px solid {bdc};border-radius:9px;padding:14px 17px')}>"
        f"<div {S(f'color:{lcolor};font-size:10px;font-weight:700;letter-spacing:1px;margin-bottom:9px')}>{label}</div>"
        f"<div {S(f'color:{TEXT};font-size:13px;line-height:1.65')}>{body}</div>"
        f"</div>"
    )

def tbl(headers, rows):
    ths = "".join(
        f"<th style='padding:7px 12px;border:1px solid {BORDER};background:#f1f5f9;"
        f"color:{PURPLE};font-size:12px;text-align:left'>{h}</th>" for h in headers)
    trs = ""
    for i, row_data in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f8faff"
        tds = "".join(
            f"<td style='padding:7px 12px;border:1px solid {BORDER};background:{bg};"
            f"font-size:12.5px;color:{TEXT}'>{c}</td>" for c in row_data)
        trs += f"<tr>{tds}</tr>"
    return f"<table style='width:100%;border-collapse:collapse'><tr>{ths}</tr>{trs}</table>"

# ── Slides ────────────────────────────────────────────────────────────────────

def s0():
    return wrap(0,
        title("📋 Problem Statement", TEAL) + "<br>" +
        q_full() + "<br>" +
        card("🔍 MAPoRL Training Flow", TEAL, TEAL_BG, TEAL,
             f"Agents <b>solve the problem independently → debate each other → improve via reinforcement learning</b>.<br>"
             f"<span style='color:{SUB}'>Press Next ▶ to walk through each step.</span>"))

def s1():
    return wrap(1,
        title("🔵 Turn 0 — Agent 0 Independent Answer", BLUE) + "<br>" +
        q_mini() + "<br>" +
        row(agent(0, T0_A0), agent(1, "Not answered yet...", faded=True)) + "<br>" +
        card("ℹ️ Turn 0 Rule", TEAL, TEAL_BG, TEAL,
             f"In Turn 0, each agent writes its solution <b>without seeing the other agent's answer</b>."))

def s2():
    return wrap(2,
        title("🔵 Turn 0 — Agent 1 Independent Answer", BLUE) + "<br>" +
        q_mini() + "<br>" +
        row(agent(0, T0_A0), agent(1, T0_A1)) + "<br>" +
        card("ℹ️ The two agents gave different answers", TEAL, TEAL_BG, TEAL,
             f"Agent 0: <b style='color:{GREEN}'>$18</b> &nbsp;|&nbsp; "
             f"Agent 1: <b style='color:{RED}'>$26</b><br>"
             f"<span style='color:{SUB}'>The Verifier will score each response for correctness.</span>"))

def s3():
    return wrap(3,
        title("⚖️ Verifier Scoring — Turn 0", AMBER) + "<br>" +
        row(agent(0, T0_A0, score=0.91, ok=True), agent(1, T0_A1, score=0.18, ok=False)) + "<br>" +
        card("⚖️ VERIFIER SCORE (Turn 0)", AMBER, AMBER_BG, AMBER,
             tbl(["Agent", "Answer", "Score"],
                 [["Agent 0", f"<b style='color:{GREEN}'>$18 ✓</b>",
                   f"<b style='color:{GREEN}'>0.91</b>"],
                  ["Agent 1", f"<b style='color:{RED}'>$26 ✗</b>",
                   f"<b style='color:{RED}'>0.18</b>"]])))

def s4():
    return wrap(4,
        title("🟢 Turn 1 — Agent 0 Debate", GREEN) + "<br>" +
        card("📨 Turn 1 Context (information given to Agent 0)", TEAL, TEAL_BG, TEAL,
             f"Agent 0 has seen Agent 1's answer: <b style='color:{RED}'>$26</b>.") + "<br>" +
        row(agent(0, T1_A0), agent(1, "Composing reply...", faded=True)))

def s5():
    return wrap(5,
        title("🟢 Turn 1 — Agent 1 Self-Correction", GREEN) + "<br>" +
        card("📨 Turn 1 Context (information given to Agent 1)", TEAL, TEAL_BG, TEAL,
             f"Agent 1 has seen Agent 0's answer <b style='color:{GREEN}'>$18</b> and its reasoning.") + "<br>" +
        row(agent(0, T1_A0), agent(1, T1_A1)))

def s6():
    return wrap(6,
        title("⚖️ Verifier Scoring — Turn 1", AMBER) + "<br>" +
        row(agent(0, T1_A0, score=0.89, ok=True),
            agent(1, T1_A1, score=0.85, ok=True, bonus=0.5)) + "<br>" +
        card("⚖️ VERIFIER SCORE (All Turns)", AMBER, AMBER_BG, AMBER,
             tbl(["Turn", "Agent 0", "Agent 1"],
                 [["Turn 0",
                   f"<span style='color:{GREEN}'>0.91 ✓</span>",
                   f"<span style='color:{RED}'>0.18 ✗</span>"],
                  ["Turn 1",
                   f"<span style='color:{GREEN}'>0.89 ✓</span>",
                   f"<span style='color:{GREEN}'>0.85 ✓</span> "
                   f"<span style='color:{PURPLE}'>(+0.5 bonus)</span>"]])))

def s7():
    return wrap(7,
        title("🎁 Bonus Rule (α₁ — Persuasion Acceptance)", PURPLE) + "<br>" +
        card("BONUS RULE", PURPLE, PURPLE_BG, PURPLE,
             tbl(["Answer(t)", "Answer(t+1)", "Majority(t)", "Bonus"],
                 [[f"<span style='color:{RED}'>Wrong (W)</span>",
                   f"<b style='color:{GREEN}'>Correct (R)</b>",
                   f"<b style='color:{GREEN}'>Correct (R)</b>",
                   f"<b style='color:{PURPLE}'>+α₁ ← this case!</b>"],
                  [f"<span style='color:{SUB}'>Wrong (W)</span>",
                   f"<span style='color:{SUB}'>Wrong (W)</span>",
                   f"<span style='color:{SUB}'>Wrong (W)</span>",
                   f"<span style='color:{SUB}'>+α₀</span>"],
                  [f"<span style='color:{SUB}'>Correct (R)</span>",
                   f"<span style='color:{SUB}'>Wrong (W)</span>",
                   f"<span style='color:{SUB}'>Correct (R)</span>",
                   f"<span style='color:{SUB}'>−α₁</span>"]]) +
             f"<br>Agent 1: Turn 0 <b style='color:{RED}'>wrong ($26)</b> → "
             f"Turn 1 <b style='color:{GREEN}'>correct ($18)</b>, "
             f"majority answer was correct → <b style='color:{PURPLE}'>+α₁ = +0.5 bonus awarded</b><br>"
             f"<span style='color:{SUB};font-size:12px'>α = [0.5, 0.5, 0.3, 0.3] (α₀, α₁, β₀, β₁)</span>"))

def s8():
    return wrap(8,
        title("💰 Final Reward Calculation (Discounted Sum, γ=0.3)", PURPLE) + "<br>" +
        card("FINAL REWARD", PURPLE, PURPLE_BG, PURPLE,
             tbl(["Agent", "Turn", "Base Score", "Bonus", "Discounted Reward"],
                 [["Agent 0", "Turn 0", "0.91", "—",
                   f"<b>0.91 + 0.3×0.89 = 1.177</b>"],
                  ["Agent 0", "Turn 1", "0.89", "—", "<b>0.89</b>"],
                  ["Agent 1", "Turn 0", "0.18", "—",
                   "<b>0.18 + 0.3×(0.85+0.5) = 0.585</b>"],
                  ["Agent 1", "Turn 1", "0.85",
                   f"<b style='color:{PURPLE}'>+0.5</b>",
                   f"<b style='color:{PURPLE}'>1.35</b>"]]) +
             f"<br><span style='color:{SUB};font-size:12px'>"
             f"Reward at turn t = score(t) + γ·score(t+1) + γ²·score(t+2) + … + bonus</span>"))

def s9():
    return wrap(9,
        title("🔄 PPO Update", TEAL) + "<br>" +
        card("POLICY UPDATE — LoRA Adapter Weight Update", TEAL, TEAL_BG, TEAL,
             tbl(["Component", "Role"],
                 [["pg_loss (Policy Gradient)",
                   f"Increase probability of responses with high reward <b style='color:{GREEN}'>↑</b>"],
                  ["vf_loss (Value Function)",
                   f"Improve accuracy of predicting next turn's reward <b style='color:{GREEN}'>↑</b>"],
                  ["KL Penalty",
                   "Prevent the policy from drifting too far from the reference model"]]) +
             f"<br>"
             f"<span style='color:{GREEN}'>✅</span> In the next episode, Agent 1 will learn to subtract the 4 muffin eggs from the start.<br>"
             f"<span style='color:{GREEN}'>✅</span> Better collaboration leads to higher scores across all turns.<br><br>"
             f"<b style='color:{TEAL}'>Repeating this episode hundreds of times = MAPoRL training</b>"))

SLIDES = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]
N = len(SLIDES)

def build(step): return SLIDES[max(0, min(step, N-1))]()

# ── Handlers ──────────────────────────────────────────────────────────────────

def go_next(step):
    step = min(step + 1, N - 1)
    return build(step), step, gr.update(interactive=step > 0), gr.update(interactive=step < N-1)

def go_prev(step):
    step = max(step - 1, 0)
    return build(step), step, gr.update(interactive=step > 0), gr.update(interactive=step < N-1)

def go_reset(_):
    return build(0), 0, gr.update(interactive=False), gr.update(interactive=True)

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MAPoRL Animation") as demo:
    gr.Markdown("# MAPoRL Training Walkthrough")
    gr.Markdown(
        "**Multi-Agent Post-Co-Training with RL** — "
        "Multiple LLMs collaborate, debate, and improve their cooperative ability through reinforcement learning.\n\n"
        "Use the **Prev / Next** buttons to step through each slide.")

    state = gr.State(0)
    html  = gr.HTML(build(0))

    with gr.Row():
        btn_prev  = gr.Button("◀ Prev",  size="sm", interactive=False)
        btn_next  = gr.Button("Next ▶",  size="sm", variant="primary")
        btn_reset = gr.Button("↺ Reset", size="sm")

    outs = [html, state, btn_prev, btn_next]
    btn_prev.click( go_prev,  inputs=[state], outputs=outs)
    btn_next.click( go_next,  inputs=[state], outputs=outs)
    btn_reset.click(go_reset, inputs=[state], outputs=outs)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
