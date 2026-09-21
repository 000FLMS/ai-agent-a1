"""The Part 3 chess agent: reuse the shared loop with one new domain tool."""

from __future__ import annotations

import json
from typing import Any

import httpx

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
)
from assignment.agent.chess_tools import (
    _game_state,
    _invoke_skill,
    _play_move,
    CHESS_PORT,
    _run_python,
    _simulate_move,
)
from assignment.agent.tools import (
    INVOKE_SKILL_TOOL,
    PLAY_MOVE_TOOL,
    RUN_PYTHON_TOOL,
    SIMULATE_MOVE_TOOL,
)
from assignment.prompts import (
    CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE,
    CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE,
    PROGRAMMATIC_CHESS_PROMPT,
)
from assignment.env import Environment


def format_chess_state(
    state: dict[str, Any], *, include_legal_moves: bool = True
) -> str:
    """Turn chess API JSON into a compact observation an LLM can act on."""

    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")

    recent_history = state.get("history", [])[-8:]
    history = (
        " ".join(
            f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history
        )
        or "(none)"
    )
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    board_text = "\n".join(board_lines)

    observation = (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n"
        f"{board_text}\n"
        f"recent_history: {history}\n"
    )
    if include_legal_moves:
        observation += f"legal_moves: {legal_moves}\n"
    return observation + "</chess_state>"


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 200,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        http_client: Any | None = None,
        reset_game: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None,
        programmatic_tools: bool = False,
        python_sandbox_port: int = CHESS_PORT,
        include_legal_moves: bool = True,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )

        # (Part 3): Register the play_move tool schema from tools.py.

        self.tools += [PLAY_MOVE_TOOL]
        self.tool_registry.update({
            PLAY_MOVE_TOOL["function"]['name']: _play_move,
        })

        if programmatic_tools:
            self.tools += [RUN_PYTHON_TOOL, SIMULATE_MOVE_TOOL]
            self.tool_registry.update({
                RUN_PYTHON_TOOL["function"]['name']: _run_python,
                SIMULATE_MOVE_TOOL["function"]['name']: _simulate_move,
            })
            if self.skills:
                # base agent already registered, overwrite it 
                self.tool_registry[INVOKE_SKILL_TOOL["function"]['name']] = _invoke_skill

        # run_python always executes in the sandbox, on the port the chess
        # server is listening on there.
        self.python_sandbox_port = python_sandbox_port
        self.include_legal_moves = include_legal_moves

        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError(
                    "ChessAgent needs an environment with server_url or an http_client."
                )
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client

        initial_state = _game_state(self.chess_client, reset=reset_game)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        prompt_template = (
            CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE
            if include_legal_moves
            else CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE
        )
        self.system_prompt = prompt_template.render()
        if programmatic_tools:
            self.system_prompt += "\n\n" + PROGRAMMATIC_CHESS_PROMPT
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )
        opening_instruction = (
            "Choose one move from legal_moves and call play_move."
            if include_legal_moves
            else "Infer a legal move from the board and FEN, then call play_move."
        )
        self.task_prompt = (
            f"Play this game as White. {opening_instruction}\n\n"
            f"{self.format_state(initial_state)}"
        )

    def format_state(self, state: dict[str, Any]) -> str:
        """Format a state according to this run's observation condition."""

        return format_chess_state(state, include_legal_moves=self.include_legal_moves)

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute model-generated ``play_move`` calls against the chess API."""

        # (Part 3.1):
        # 1. Dispatch on the function name, and ignore a tool this agent did
        #    not register.
        # 2. Hand the raw arguments to the matching chess_tools helper, with
        #    self.chess_client as its first argument. Each helper takes the
        #    client explicitly so the same code can run inside the sandbox.
        # 3. Format a played move with self.format_state, then update
        #    last_state and finished.
        # 4. Link every observation to its call with tool_call_id.
        # 5. Turn malformed, unknown, rejected, or extra parallel calls into
        #    recoverable <chess_error> observations instead of crashing.

        obvs = []
        
        for call in tool_calls:
            tool_name = call.get("function", {}).get("name", "unknown")
            tool_id =  call.get("id", "")
            obv = {
                'role': 'tool', 
                'content': '', 
                'tool_call_id': tool_id, 
                'name': tool_name
            }
            try:
                args = call.get("function", {}).get("arguments", "")
                
                if tool_name in self.tool_registry:
                    if tool_name == RUN_PYTHON_TOOL['function']["name"]:
                        result = self.tool_registry[tool_name](self.env, self.python_sandbox_port, args)
                    elif tool_name == INVOKE_SKILL_TOOL['function']["name"]:
                        result = self.tool_registry[tool_name](self.skills, args)
                    else:
                        result = self.tool_registry[tool_name](self.chess_client, args)

                    if tool_name != INVOKE_SKILL_TOOL['function']["name"] and "<chess_error>" not in result:
                        if tool_name == PLAY_MOVE_TOOL['function']["name"]:
                            state = json.loads(result)
                            obv["content"] = self.format_state(state)
                        else:
                            state = _game_state(self.chess_client)
                            obv["content"] = f"Execute result: {result} \n Chess state: {self.format_state(state)}"
                            
                        self.last_state = state
                        self.finished = bool(state.get("game_over"))
                    else:
                        obv["content"] = result 
                    
                else:
                    obv["content"] = "<chess_error>  ERROR: Called unknown tool </chess_error>"

            except Exception as e:
                obv["content"] = f"<chess_error> {str(e)} </chess_error>"


            obvs.append(obv)

        return obvs



        
