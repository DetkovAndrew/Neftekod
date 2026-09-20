"""
Шина сообщений одного цикла принятия решения (ARCHITECTURE.md §6).

Агенты вызываются Оркестратором синхронно и детерминированно, но каждый
обмен явно фиксируется как сообщение отправитель -> получатель с типом
полезной нагрузки. Трасса попадает в Recommendation.trace и в журнал
цикла (runs/<ts>/00_trace.json): по ней видно, кто что кому передал и
на каком шаге цикл остановился (отказ, Guard BLOCK, отсутствие риска).
"""

from __future__ import annotations

from neftekod_mas.schemas import AgentMessage


class MessageBus:
    def __init__(self) -> None:
        self.messages: list[AgentMessage] = []

    def send(self, sender: str, recipient: str, topic: str, summary: str) -> None:
        self.messages.append(
            AgentMessage(seq=len(self.messages) + 1, sender=sender, recipient=recipient, topic=topic, summary=summary)
        )
