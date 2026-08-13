"""AI 中文讲解功能：基于 KataGo 分析结果生成自然语言讲解，并提供候选点对比、变化推演和目数对比。"""
import json
import threading
from typing import Callable, Dict, List, Optional

from katrain.core.lang import i18n
from katrain.core.sgf_parser import Move


PLAYER_NAMES = {"B": i18n._("Black"), "W": i18n._("White")}

# 用于标注候选点的字母（排除容易和坐标/数字混淆的）
CANDIDATE_LETTERS = "ABCDEFGHJKLMN"


def _fmt_score(score_lead: float, player: str) -> str:
    """把黑方视角的 scoreLead 格式化成某方领先目数。"""
    sign = 1 if player == "B" else -1
    val = sign * score_lead
    if val >= 0:
        return f"{PLAYER_NAMES[player]}{i18n._('leads by')}{val:.1f}{i18n._('points')}"
    return f"{PLAYER_NAMES[player]}{i18n._('trails by')}{-val:.1f}{i18n._('points')}"


def _fmt_winrate(winrate: float, player: str) -> float:
    """黑方视角 winrate -> 当前玩家视角百分比。"""
    wr = winrate if player == "B" else 1 - winrate
    return wr * 100


def _synchronous_analysis(engine, node, visits: int, ownership: bool = True,
                           find_alternatives: bool = False) -> Dict:
    """发起一次分析并阻塞等待最终结果，返回 KataGo 的 analysis dict。"""
    result = {"data": None, "error": None}
    event = threading.Event()

    def callback(analysis, partial):
        if partial:
            return
        result["data"] = analysis
        event.set()

    def error_callback(msg, *args):
        result["error"] = msg
        event.set()

    engine.request_analysis(
        node,
        callback,
        error_callback=error_callback,
        visits=visits,
        ownership=ownership,
        find_alternatives=find_alternatives,
        priority=100,
        time_limit=False,
    )
    event.wait(timeout=120)
    return result


def _player_to_move_text(node) -> str:
    return PLAYER_NAMES[node.next_player]


def _describe_move(move_gtp: str, player: str) -> str:
    if move_gtp == "pass":
        return f"{PLAYER_NAMES[player]}{i18n._('passes')}"
    return f"{PLAYER_NAMES[player]} {move_gtp}"


def _describe_territory(ownership_grid, size_x: int, size_y: int, player: str) -> str:
    """根据 ownership 粗略估算黑白双方的"势力目数"。ownership 为黑方视角 -1~1。"""
    black_pts = 0.0
    white_pts = 0.0
    for y in range(size_y):
        for x in range(size_x):
            v = ownership_grid[y][x]
            if v > 0:
                black_pts += v
            elif v < 0:
                white_pts += -v
    return (f"{i18n._('Black territory estimate')}: {black_pts:.1f}, "
            f"{i18n._('White territory estimate')}: {white_pts:.1f}")


def generate_explanation(katrain, node, visits: int = 200, num_candidates: int = 4,
                         on_update: Optional[Callable[[str], None]] = None) -> Dict:
    """对 node 局面生成讲解。

    on_update: 流式更新回调，接收一段进度文字。
    返回 dict: {text, candidates:[{letter,move,score,winrate,pointsLost,pv}], chosen_index, pv, territory}
    """
    engine = katrain.engine
    size_x, size_y = node.board_size
    player = node.next_player

    def say(msg):
        if on_update:
            on_update(msg)

    say(i18n._("Analyzing current position..."))

    # 第一次分析：拿到候选点
    r1 = _synchronous_analysis(engine, node, visits=visits, ownership=True)
    if r1["error"]:
        return {"error": r1["error"], "text": i18n._("Engine error: {msg}").format(msg=r1["error"])}

    analysis = r1["data"]
    if not analysis or "moveInfos" not in analysis:
        return {"error": "no result", "text": i18n._("No analysis available")}

    root = analysis.get("rootInfo", {})
    move_infos = sorted(analysis.get("moveInfos", []), key=lambda m: m.get("order", 99))
    top_moves = move_infos[:num_candidates]

    if not top_moves:
        return {"text": i18n._("No candidate moves found.")}

    # 候选点标注字母
    for idx, m in enumerate(top_moves):
        m["letter"] = CANDIDATE_LETTERS[idx] if idx < len(CANDIDATE_LETTERS) else str(idx + 1)

    best = top_moves[0]
    best_score = best["scoreLead"]
    best_wr = _fmt_winrate(best["winrate"], player)

    lines = []
    lines.append(f"[b]{i18n._('AI Explanation')}[/b]")
    lines.append("")
    lines.append(f"[b]{_player_to_move_text(node)}{i18n._('to move')}[/b]，"
                 f"{i18n._('overall')}：{_fmt_score(root.get('scoreLead', 0), 'B')}，"
                 f"{i18n._('winrate')}：{_fmt_winrate(root.get('winrate', 0.5), 'B'):.1f}% / "
                 f"{_fmt_winrate(root.get('winrate', 0.5), 'W'):.1f}%")
    lines.append("")

    # 最佳点
    lines.append(f"[b][color=#9bd46b]★ {i18n._('Recommended move')}: {best['move']} "
                 f"({i18n._('letter')} {best['letter']})[/color][/b]")
    lines.append(f"  · {i18n._('score lead')}: {_fmt_score(best_score, 'B')}（"
                 f"{i18n._('for')}{PLAYER_NAMES[player]}{i18n._('perspective')}："
                 f"{'+' if (1 if player=='B' else -1)*best_score>=0 else ''}"
                 f"{((1 if player=='B' else -1)*best_score):.1f}{i18n._('points')}）")
    lines.append(f"  · {i18n._('winrate')}: {best_wr:.1f}%（{PLAYER_NAMES[player]}）；"
                 f"{i18n._('visits')}: {best.get('visits', 0)}")
    if best.get("prior") is not None:
        lines.append(f"  · {i18n._('policy prior')}: {best['prior']*100:.1f}%")

    # 主变化推演
    pv = best.get("pv", [])[:8]
    if pv:
        pv_str = " → ".join(pv)
        lines.append("")
        lines.append(f"[b]{i18n._('Main variation')}（PV）:[/b]")
        lines.append(f"  {best['move']} → {pv_str}")

    # 候选点对比
    if len(top_moves) > 1:
        lines.append("")
        lines.append(f"[b]{i18n._('Candidate comparison')}（{i18n._('points/winrate relative to best')}）:[/b]")
        for m in top_moves[1:]:
            delta = best_score - m["scoreLead"]  # 黑方视角
            delta_for_player = (1 if player == "B" else -1) * delta  # 正数=对当前玩家不利
            wr_delta = _fmt_winrate(best["winrate"], player) - _fmt_winrate(m["winrate"], player)
            marker = "⚠" if abs(delta_for_player) >= 1 else "·"
            direction = i18n._('worse') if delta_for_player > 0 else i18n._('better')
            lines.append(
                f"  {marker} [{m['letter']}] {m['move']}: "
                f"{abs(delta_for_player):.1f}{i18n._('points')}{direction}，"
                f"{i18n._('winrate')} {wr_delta:+.1f}%，"
                f"{i18n._('visits')} {m.get('visits',0)}"
            )
            alt_pv = m.get("pv", [])[:5]
            if alt_pv:
                lines.append(f"      {i18n._('variation')}: {m['move']} → {' → '.join(alt_pv)}")

    # "为什么不是那里"——和实际落子对比
    actual_move = node.move.gtp() if node.move else None
    if actual_move and node.parent and node.parent.analysis_exists:
        parent_candidates = node.parent.candidate_moves
        if parent_candidates:
            best_at_parent = parent_candidates[0]["move"]
            if actual_move != best_at_parent:
                points_lost = node.points_lost
                lines.append("")
                lines.append(f"[b][color=#e88]⚑ {i18n._('Your move')}: {actual_move}[/color][/b]")
                lines.append(f"  · {i18n._('AI recommended')} {best_at_parent}，"
                             f"{i18n._('you lost about')} {points_lost:.1f}{i18n._('points')}")
                if actual_move in [m["move"] for m in move_infos]:
                    am = next(m for m in move_infos if m["move"] == actual_move)
                    lines.append(f"  · {i18n._('Your winrate')}: "
                                 f"{_fmt_winrate(am['winrate'], player):.1f}%，"
                                 f"{i18n._('best winrate')}: {best_wr:.1f}%")

    # 目数/势力对比（ownership）
    ownership = analysis.get("ownership")
    territory_text = ""
    if ownership:
        try:
            from katrain.core.utils import var_to_grid
            grid = var_to_grid(ownership, (size_x, size_y))
            territory_text = _describe_territory(grid, size_x, size_y, player)
            lines.append("")
            lines.append(f"[b]{i18n._('Territory estimate')}（{i18n._('based on ownership')}）:[/b]")
            lines.append(f"  {territory_text}")
        except Exception:
            pass

    lines.append("")
    lines.append(f"[size=12][color=#aaa]{i18n._('Analysis visits')}: {root.get('visits', 0)} · "
                 f"{i18n._('Tip')}: {i18n._('Click a candidate letter on the board to play its variation.')}[/color][/size]")

    text = "\n".join(lines)

    # 用 LLM 生成自然语言讲解（若已配置）
    llm_text = ""
    try:
        from katrain.core.llm import is_configured, chat_completion, LLMError, get_model_display_name
        if is_configured(katrain) and katrain.config("llm/use_llm", True):
            say(i18n._("Generating natural language explanation..."))
            prompt = _build_llm_prompt(node, root, top_moves, actual_move)
            llm_text = chat_completion(
                katrain,
                prompt,
                system_prompt=(
                    "你是一位围棋九段职业棋手，擅长用简洁、易懂的中文向业余爱好者讲解棋局。"
                    "请基于我提供的 KataGo 分析数据（候选点、胜率、目数、主变化），"
                    "解释为什么推荐下在这里、而不是其它候选点，语言自然、有围棋味道，"
                    "不要罗列数字，要像老师面对面讲棋一样。控制在 200 字以内。"
                ),
            )
    except Exception as e:
        llm_text = f"[color=#e88]{i18n._('LLM explanation failed')}: {e}[/color]"

    if llm_text:
        text += f"\n\n[b]{i18n._('AI Natural Language Explanation')}（{get_model_display_name(katrain)}）:[/b]\n{llm_text}"

    return {
        "text": text,
        "candidates": top_moves,
        "best_move": best["move"],
        "pv": pv,
        "territory": territory_text,
    }


def _build_llm_prompt(node, root, top_moves, actual_move: Optional[str]) -> str:
    """构造喂给 LLM 的 prompt（JSON 结构化数据 + 少量上下文）。"""
    size_x, size_y = node.board_size
    player = node.next_player
    payload = {
        "board_size": f"{size_x}x{size_y}",
        "komi": node.komi,
        "rules": node.ruleset,
        "player_to_move": "Black" if player == "B" else "White",
        "current_score_lead_black": root.get("scoreLead", 0),
        "current_winrate_black": root.get("winrate", 0.5),
        "actual_move_played": actual_move,
        "candidates": [
            {
                "move": m["move"],
                "score_lead_black": m["scoreLead"],
                "winrate_black": m["winrate"],
                "points_lost_vs_best": round(
                    (top_moves[0]["scoreLead"] - m["scoreLead"]) * (1 if player == "B" else -1), 2
                ),
                "winrate_lost_vs_best": round(
                    (top_moves[0]["winrate"] - m["winrate"]) * (1 if player == "B" else -1), 4
                ),
                "visits": m.get("visits", 0),
                "policy_prior": round(m.get("prior", 0), 4),
                "pv": m.get("pv", [])[:6],
            }
            for m in top_moves
        ],
    }
    return (
        "请根据下面的 KataGo 分析 JSON，为当前局面写一段自然语言讲解：\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n"
        "要求：1) 用中文；2) 先给出推荐点并解释为什么；3) 简要对比 2~3 个候选点的优劣；"
        "4) 如果实际落子不是最佳，点明亏了多少目；5) 提及主变化的后续思路。"
    )


def build_variation_branch(node, move_gtp: str, length: int = 8):
    """在 node 下创建一个变化分支，按候选点的 PV 摆 length 手。返回分支末端节点（不切换当前节点）。"""
    player = node.next_player
    first = Move.from_gtp(move_gtp, player=player)
    child = node.play(first)
    cur = child
    # PV 中已经包含了 first move 之后的后续
    return child
