"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
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
        self.task = task
        self.submitted_patch = ""

        # (Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools += [EXECUTE_TOOL, SEND_MESSAGE_TOOL]
        self.tool_registry.update({
            'execute': self.env.execute,
            "send_message": self._send_message
        })

        # (1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.

        self.sys_info = {
            "machine": self.env.machine,
            "release": self.env.release,
            "system": self.env.system,
            "version": self.env.version
        }

        self.system_prompt = f"""
        You are a professional coding agent, you can only work inside this system with provided tools:
        <system_information>
            {self.sys_info}
        </system_information>"""

        self.task_prompt = f"""
        Here is one specific coding task you need to complete:
        <task_description>
            {self.task}
        </task_description>"""

        # (1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            skill_prompt = [{'skill_name': name, 'metadata': info['metadata']} for name, info in self.skills.items()]
            self.system_prompt += f"""\n You can also use these SKILLs: 
                        <skills>
                            {json.dumps(skill_prompt)}
                        </skills>\n"""
            print(self.system_prompt)


    def _send_message(self, summary:str) -> str:
        print(f"CodeAgent: {summary}\n")
        return "Message sent to user successfully"


    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # (Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.

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
                args = json.loads(call.get("function", {}).get("arguments", {}))
                if tool_name in self.tool_registry:
                    result = self.tool_registry[tool_name](**args)
                    obv["content"] = (result if isinstance(result, str) else format_tool_output(result))
                else:
                    obv["content"] = "ERROR: Called unknown tool"

            except Exception as e:
                obv["content"] = str(e)

            obvs.append(obv)

        return obvs
