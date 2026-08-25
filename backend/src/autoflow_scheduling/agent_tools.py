from __future__ import annotations

from typing import Final

CUSTOMER_AGENT_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "get_work_order_status",
        "create_work_order",
        "cancel_work_order",
        "reschedule_work_order",
    }
)

STORE_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "check_in_work_order",
        "complete_work_order",
    }
)


def customer_agent_can_use(tool_name: str) -> bool:
    return tool_name in CUSTOMER_AGENT_TOOLS
