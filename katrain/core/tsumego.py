"""死活题模式：加载题库、做题、KataGo 判对错、AI 讲解。"""
import json
import os
import re
import random
from typing import Optional, Dict, Any, List, Tuple

from katrain.core.constants import (
    DATA_FOLDER,
    MODE_TSUMEGO,
    OUTPUT_INFO,
    OUTPUT_ERROR,
    STATUS_INFO,
    STATUS_ERROR,
)
from katrain.core.lang import i18n
from katrain.core.game_node import GameNode
from katrain.core.sgf_parser import SGF, SGFNode, Move


def _synchronous_analysis(engine, node, visits=100, timeout=30):
    """阻塞式分析，返回结果字典。"""
    import threading
    result_holder = {}
    done = threading.Event()

    def callback(result):
        result_holder["result"] = result
        done.set()

    def error_callback(error):
        result_holder["error"] = error
        done.set()

    engine.request_analysis(
        node,
        callback=callback,
        error_callback=error_callback,
        visits=visits,
        time_limit=False,
    )
    done.wait(timeout)
    return result_holder.get("result")


class TsumegoProblem:
    """单个死活题。"""

    def __init__(self, sgf_node: SGFNode, source_file: str, problem_id: str):
        self.sgf_node = sgf_node
        self.source_file = source_file
        self.problem_id = problem_id
        self.name = self._extract_name()
        self.to_play = self._extract_to_play()
        self.setup_stones = self._extract_setup()

    def _extract_name(self) -> str:
        comment = self.sgf_node.get_property("C", "")
        m = re.search(r"problem\s+(\d+)", comment)
        if m:
            return f"{os.path.basename(self.source_file)} #{m.group(1)}"
        return f"{os.path.basename(self.source_file)} - {self.problem_id}"

    def _extract_to_play(self) -> str:
        return self.sgf_node.get_property("PL", "B").upper()

    def _extract_setup(self) -> List[Tuple[str, str]]:
        """返回 [(color, coord), ...] 列表。"""
        stones = []
        for color, prop in [("B", "AB"), ("W", "AW")]:
            for coord in self.sgf_node.get_list_property(prop, []):
                stones.append((color, coord))
        return stones


class TsumegoLibrary:
    """题库管理器。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tsumego_library")
        self.library_path = os.path.normpath(self.library_path)
        self.index_path = os.path.join(self.library_path, "index.json")
        self.categories: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded = False

    def load_index(self):
        """加载索引文件。"""
        if self._loaded:
            return
        if not os.path.exists(self.index_path):
            self._build_index()
        with open(self.index_path) as f:
            data = json.load(f)
        self.categories = data.get("categories", {})
        self._loaded = True

    def _build_index(self):
        """扫描 SGF 文件构建索引。"""
        categories = {"elementary": [], "intermediate": [], "advanced": []}
        classical_dir = os.path.join(self.library_path, "classical")
        if not os.path.isdir(classical_dir):
            self.categories = categories
            return

        # 按文件名映射难度
        difficulty_map = {
            "cho-1.sgf": "elementary",
            "cho-2.sgf": "intermediate",
            "cho-3.sgf": "advanced",
            "lee-chang-ho.sgf": "intermediate",
            "gokyoshumyo.sgf": "advanced",
            "hatsuyoron.sgf": "advanced",
            "xxqj.sgf": "advanced",
        }

        for fname, difficulty in difficulty_map.items():
            path = os.path.join(classical_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                content = f.read()
            # 提取 problem 分支
            for m in re.finditer(r"\(;C\[problem (\d+)\]([^\)]*)\)", content):
                num = m.group(1)
                props = m.group(2)
                categories[difficulty].append({
                    "file": fname,
                    "id": f"{fname[:-4]}-{num}",
                    "name": f"{fname[:-4]} #{num}",
                    "props": props,
                })

        self.categories = categories
        with open(self.index_path, "w") as f:
            json.dump({"categories": categories}, f, indent=2, ensure_ascii=False)

    def get_random_problem(self, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """随机抽一道题。"""
        self.load_index()
        if category and category in self.categories:
            pool = self.categories[category]
        else:
            pool = []
            for probs in self.categories.values():
                pool.extend(probs)
        if not pool:
            return None
        return random.choice(pool)

    def get_problem_by_id(self, problem_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 找题。"""
        self.load_index()
        for probs in self.categories.values():
            for p in probs:
                if p["id"] == problem_id:
                    return p
        return None

    def load_problem(self, problem_info: Dict[str, Any]) -> Optional[TsumegoProblem]:
        """加载单个题目为 TsumegoProblem。"""
        path = os.path.join(self.library_path, "classical", problem_info["file"])
        if not os.path.exists(path):
            return None

        with open(path) as f:
            content = f.read()

        # 找到对应 problem 分支
        num = problem_info["id"].split("-")[-1]
        pattern = r"\(;C\[problem " + num + r"\]([^\)]*)\)"
        m = re.search(pattern, content)
        if not m:
            return None

        # 构造一个 SGFNode
        props_str = m.group(1)
        node = SGFNode(properties=SGF.parse_properties(props_str))
        return TsumegoProblem(node, problem_info["file"], problem_info["id"])


class TsumegoSession:
    """一次做题会话。"""

    def __init__(self, katrain, problem: TsumegoProblem):
        self.katrain = katrain
        self.problem = problem
        self.current_node: Optional[GameNode] = None
        self.attempts = 0
        self.solved = False
        self.failed = False
        self._setup_board()

    def _setup_board(self):
        """把题目摆到棋盘上。"""
        # 创建新游戏节点
        root = GameNode(self.katrain.game, properties={
            "SZ": ["19"],
            "RU": ["Chinese"],
            "KM": ["7.5"],
        })
        for color, coord in self.problem.setup_stones:
            root.add_list_property("AB" if color == "B" else "AW", [coord])
        root.set_property("PL", [self.problem.to_play])
        self.katrain.game.root = root
        self.katrain.game.current_node = root
        self.current_node = root
        self.katrain.board_gui.redraw_board()
        self.katrain.log(f"Tsumego: {self.problem.name}, {self.problem.to_play} to play", OUTPUT_INFO)

    def play_move(self, coords: str) -> Tuple[bool, str]:
        """
        用户下一手，返回 (是否好棋, 反馈消息)。
        用 KataGo 分析判断。
        """
        if self.solved or self.failed:
            return False, i18n._("This problem is already finished. Load a new one.")

        self.attempts += 1

        # 创建新节点
        move = Move.from_gtp(coords, player=self.current_node.next_player)
        new_node = self.current_node.play(move)
        self.katrain.game.set_current_node(new_node)
        self.current_node = new_node

        # 分析
        analysis = self._analyze_position()
        if analysis is None:
            return False, i18n._("Analysis failed. Please try again.")

        # 判断好坏：比较当前节点和父节点的胜率/目数
        parent_analysis = self._analyze_position(self.current_node.parent)
        if parent_analysis is None:
            return False, i18n._("Analysis failed. Please try again.")

        current_score = analysis.get("rootInfo", {}).get("scoreLead", 0)
        parent_score = parent_analysis.get("rootInfo", {}).get("scoreLead", 0)
        current_winrate = analysis.get("rootInfo", {}).get("winrate", 0)
        parent_winrate = parent_analysis.get("rootInfo", {}).get("winrate", 0)

        # 对于黑先：分数下降=坏棋；对于白先：分数上升=坏棋
        is_black = self.problem.to_play == "B"
        score_delta = current_score - parent_score if is_black else parent_score - current_score
        winrate_delta = current_winrate - parent_winrate if is_black else parent_winrate - current_winrate

        # 阈值：目数损失 > 2 或胜率下降 > 3% 算坏棋
        is_good = score_delta > -2.0 and winrate_delta > -0.03

        if is_good:
            # 检查是否已解决（通过 KataGo 的 PV 或 ownership 判断）
            if self._check_solved(analysis):
                self.solved = True
                return True, i18n._("Correct! Problem solved!")
            return True, i18n._("Good move! Continue...")
        else:
            if self.attempts >= 3:
                self.failed = True
                best = self._get_best_move(parent_analysis)
                return False, i18n._(
                    f"Not the best move ({score_delta:+.1f} pts). "
                    f"AI suggests: {best}. Problem failed after 3 attempts."
                )
            return False, i18n._(
                f"Not the best move ({score_delta:+.1f} pts). Try again."
            )

    def _analyze_position(self, node: Optional[GameNode] = None) -> Optional[Dict[str, Any]]:
        """同步分析指定节点。"""
        if node is None:
            node = self.current_node
        try:
            engine = self.katrain.engine
            if engine is None:
                return None
            return _synchronous_analysis(engine, node, visits=100)
        except Exception as e:
            self.katrain.log(f"Tsumego analysis error: {e}", OUTPUT_ERROR)
            return None

    def _check_solved(self, analysis: Dict[str, Any]) -> bool:
        """检查是否解决（简化：看胜率是否 > 85% 或 < 15%，取决于哪方）。"""
        winrate = analysis.get("rootInfo", {}).get("winrate", 0.5)
        is_black = self.problem.to_play == "B"
        if is_black:
            return winrate > 0.85
        else:
            return winrate < 0.15

    def _get_best_move(self, analysis: Dict[str, Any]) -> str:
        """获取 AI 推荐的最佳点。"""
        move_infos = analysis.get("moveInfos", [])
        if not move_infos:
            return "pass"
        best = max(move_infos, key=lambda m: m.get("visits", 0))
        return best.get("move", "pass")

    def get_hint(self) -> str:
        """获取提示（AI 推荐点）。"""
        analysis = self._analyze_position()
        if analysis is None:
            return i18n._("Analysis unavailable.")
        best = self._get_best_move(analysis)
        return i18n._(f"Hint: try {best}")

    def get_explanation(self) -> str:
        """获取 AI 讲解。"""
        analysis = self._analyze_position()
        if analysis is None:
            return i18n._("Analysis unavailable.")

        root_info = analysis.get("rootInfo", {})
        move_infos = analysis.get("moveInfos", [])[:4]

        lines = [
            f"[b]{self.problem.name}[/b]",
            f"{self.problem.to_play} to play",
            "",
            f"Winrate: {root_info.get('winrate', 0):.1%}",
            f"Score: {root_info.get('scoreLead', 0):+.1f}",
            "",
            "Candidate moves:",
        ]
        for i, mi in enumerate(move_infos):
            move = mi.get("move", "?")
            wr = mi.get("winrate", 0)
            score = mi.get("scoreLead", 0)
            visits = mi.get("visits", 0)
            pv = " ".join(mi.get("pv", [])[:5])
            lines.append(f"{chr(65+i)}: {move}  {wr:.1%}  {score:+.1f}  ({visits}v)  {pv}")

        return "\n".join(lines)


# 全局会话
_current_session: Optional[TsumegoSession] = None


def start_tsumego_session(katrain, category: Optional[str] = None) -> Tuple[bool, str]:
    """开始一道死活题。"""
    global _current_session
    lib = TsumegoLibrary()
    problem_info = lib.get_random_problem(category)
    if problem_info is None:
        return False, i18n._("No problems found in library.")
    problem = lib.load_problem(problem_info)
    if problem is None:
        return False, i18n._("Failed to load problem.")
    _current_session = TsumegoSession(katrain, problem)
    return True, f"Loaded: {problem.name}"


def get_current_session() -> Optional[TsumegoSession]:
    return _current_session


def end_tsumego_session():
    global _current_session
    _current_session = None
