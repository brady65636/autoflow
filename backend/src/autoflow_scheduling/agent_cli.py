from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import uuid
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AgentCliError(RuntimeError):
    pass


@dataclass
class AgentApiClient:
    base_url: str
    access_token: str | None = None

    def login(self, username: str, password: str) -> dict:
        response = self._request(
            "POST",
            "/api/auth/token",
            {"username": username, "password": password},
        )
        self.access_token = str(response["access_token"])
        return response

    def chat(self, message: str, *, thread_id: str = "default") -> dict:
        if not self.access_token:
            raise AgentCliError("not authenticated")
        return self._request(
            "POST",
            "/api/agent/chat",
            {"message": message, "thread_id": thread_id},
        )

    def _request(self, method: str, path: str, payload: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=120) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = _read_error(error)
            raise AgentCliError(f"HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise AgentCliError(str(error)) from error
        if not isinstance(value, dict):
            raise AgentCliError("API response must be a JSON object")
        return value


def _read_error(error: HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        if isinstance(payload, dict) and payload.get("detail"):
            return str(payload["detail"])
        return json.dumps(payload, ensure_ascii=False)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error.reason or "request failed"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoFlow Agent CLI")
    parser.add_argument(
        "--base-url",
        default=os.getenv("AUTOFLOW_API_BASE_URL", "http://127.0.0.1:8000"),
        help="AutoFlow API base URL",
    )
    parser.add_argument("--username", help="Login username; prompts when omitted")
    parser.add_argument("--password", help="Avoid in shell history; prompts when omitted")
    parser.add_argument("--message", help="Send one message and exit instead of interactive mode")
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation ID; a random ID is used for interactive sessions",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    username = args.username or input("Username: ").strip()
    password = args.password or getpass.getpass("Password: ")
    client = AgentApiClient(args.base_url)
    thread_id = args.thread_id or str(uuid.uuid4())
    try:
        token = client.login(username, password)
        print(f"登录成功，角色：{token.get('role', 'unknown')}")
        if args.message:
            _print_response(client.chat(args.message, thread_id=thread_id))
            return 0
        print("输入消息开始对话；输入 /exit 或 Ctrl+C 退出。")
        while True:
            try:
                message = input("\nYou> ").strip()
            except EOFError:
                print()
                return 0
            if not message:
                continue
            if message.casefold() in {"/exit", "/quit", "exit", "quit"}:
                return 0
            _print_response(client.chat(message, thread_id=thread_id))
    except (AgentCliError, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            print()
            return 130
        print(f"错误：{error}", file=sys.stderr)
        return 1


def _print_response(response: dict) -> None:
    message = response.get("message", "")
    print(f"\nAgent> {message}")
    tool_calls = response.get("tool_calls") or []
    if tool_calls:
        print(f"[tools: {', '.join(map(str, tool_calls))}]")


if __name__ == "__main__":
    raise SystemExit(main())
