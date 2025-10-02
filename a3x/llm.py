"""Clientes de LLM utilizados pelo agente."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import httpx
import yaml

# Ollama client is imported only when needed for fallback
from .actions import ActionType, AgentAction, AgentState
from .cache import llm_cache_manager


class BaseLLMClient(ABC):
    """Interface para clientes LLM."""

    def start(self, goal: str) -> None:  # pragma: no cover - padrão vazio
        """Hook opcional para inicialização."""

    @abstractmethod
    def propose_action(self, state: AgentState) -> AgentAction:
        """Retorna a próxima ação a partir do estado atual."""

    def notify_observation(
        self, observation_text: str
    ) -> None:  # pragma: no cover - padrão vazio
        """Hook opcional após execução de uma ação."""

    def get_last_metrics(self) -> dict[str, float]:  # pragma: no cover - padrão vazio
        """Retorna métricas da última chamada.

        Implementações podem expor latência, retries etc. Padrão: sem métricas.
        """

        return {}


class ManualLLMClient(BaseLLMClient):
    """Cliente simples que lê ações pré-definidas de um arquivo YAML."""

    def __init__(self, script_path: Path | None) -> None:
        if script_path is None:
            raise ValueError("ManualLLMClient requer um caminho para script YAML")
        self.actions = list(_load_actions(Path(script_path)))
        self._index = 0
        self._last_metrics: dict[str, float] = {}

    def propose_action(self, state: AgentState) -> AgentAction:
        if self._index < len(self.actions):
            action = self.actions[self._index]
            self._index += 1
            return action
        # Sem ações restantes, finaliza automaticamente
        return AgentAction(type=ActionType.FINISH, text="Script manual concluído.")

    def get_last_metrics(self) -> dict[str, float]:  # pragma: no cover - sem métricas
        return dict(self._last_metrics)


class OpenRouterLLMClient(BaseLLMClient):
    """Cliente que consome a API da OpenRouter (ex.: Grok 4 Fast)."""

    _DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    _SYSTEM_PROMPT = (
        "Você é um agente autônomo que escreve e depura código. "
        "Analise o objetivo do usuário e o histórico (se houver). "
        "Responda sempre em JSON único com os campos apropriados para descrever a ação."
    )

    def __init__(
        self,
        model: str,
        api_key_env: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ) -> None:
        if not model:
            raise ValueError("OpenRouterLLMClient requer um nome de modelo válido")
        self.model = model
        env_name = api_key_env or "OPENROUTER_API_KEY"
        api_key = os.getenv(env_name)
        if not api_key:
            raise RuntimeError(
                f"Variável de ambiente {env_name} não encontrada; configure a API key da OpenRouter"
            )
        self.api_key = api_key
        self.base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._goal: str | None = None
        self._last_metrics: dict[str, float] = {}
        # Ollama client is only initialized when needed for fallback
        self.ollama_client: Any | None = None
        self._ollama_unavailable_reason: str | None = None

    def start(self, goal: str) -> None:
        self._goal = goal

    def propose_action(self, state: AgentState) -> AgentAction:
        if not self._goal:
            raise RuntimeError(
                "Cliente OpenRouter não inicializado; chame start(goal) antes"
            )

        messages = self._build_messages(state)

        # Try cache first
        llm_cache = llm_cache_manager.get_cache("openrouter_responses")
        cached_response = llm_cache.get(self.model, messages, temperature=0.1)

        if cached_response:
            content = cached_response.get_content()
            self._last_metrics["cache_hit"] = 1.0
            return self._content_to_action(content)

        # Cache miss - make API call
        response = self._send_request(messages)
        content = self._extract_content(response)

        # Cache the response
        usage = response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})
        finish_reason = response.get("choices", [{}])[0].get("finish_reason", "completed")
        llm_cache.put(self.model, messages, response, usage, finish_reason, temperature=0.1)

        self._last_metrics["cache_hit"] = 0.0
        return self._content_to_action(content)

    # Internos -----------------------------------------------------------------

    def _build_messages(self, state: AgentState) -> list[dict[str, str]]:
        history = state.history_snapshot or "(sem histórico disponível)"
        instruction = (
            "Formate a resposta como JSON com os campos: type, text, command, cwd, diff, path, content. "
            "Use apenas os campos necessários. 'type' deve ser um dos: "
            "message, run_command, apply_patch, write_file, read_file, finish."
        )
        lessons_block = (state.memory_lessons or "").strip()
        context_text = (state.seed_context or "").strip()
        if lessons_block and lessons_block in context_text:
            context_text = context_text.replace(lessons_block, "").strip()
        context_section = context_text or "Sem dados SeedAI prévios disponíveis."

        sections = [
            f"Objetivo: {self._goal}",
            f"Iteração atual: {state.iteration} / {state.max_iterations}",
            f"Contexto SeedAI:\n{context_section}",
        ]

        if lessons_block:
            sections.append(lessons_block)

        sections.append(f"Historico:\n{history}")
        sections.append(instruction)

        user_content = "\n\n".join(sections)
        return [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _send_request(self, messages: list[dict[str, str]]) -> dict:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "A3X Agent",
        }
        attempt = 0
        delay = 1.0
        started_at = time.perf_counter()
        while True:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    self._last_metrics = {
                        "llm_latency": time.perf_counter() - started_at,
                        "llm_retries": float(attempt),
                    }
                    raise RuntimeError(
                        f"Falha ao conectar com a OpenRouter: {exc}"
                    ) from exc
                time.sleep(delay)
                delay *= self.retry_backoff
                continue
            except httpx.HTTPError as exc:
                self._last_metrics = {
                    "llm_latency": time.perf_counter() - started_at,
                    "llm_retries": float(attempt),
                }
                raise RuntimeError(
                    f"Falha ao conectar com a OpenRouter: {exc}"
                ) from exc

            if response.status_code >= 500 and attempt < self.max_retries:
                attempt += 1
                time.sleep(delay)
                delay *= self.retry_backoff
                continue

            if response.status_code == 429:
                self._last_metrics = {
                    "llm_latency": time.perf_counter() - started_at,
                    "llm_retries": float(attempt),
                    "llm_status_code": float(response.status_code),
                    "llm_fallback_used": 1.0,
                }
                if not self._ensure_ollama_client():
                    reason = self._ollama_unavailable_reason or "motivo desconhecido"
                    self._last_metrics["ollama_error"] = 1.0
                    raise RuntimeError(
                        "OpenRouter retornou 429 e fallback Ollama indisponível: "
                        f"{reason}"
                    )
                return self._send_with_ollama(messages)

            if response.status_code >= 400:
                self._last_metrics = {
                    "llm_latency": time.perf_counter() - started_at,
                    "llm_retries": float(attempt),
                    "llm_status_code": float(response.status_code),
                }
                raise RuntimeError(
                    f"Erro da OpenRouter ({response.status_code}): {response.text.strip()}"
                )
            self._last_metrics = {
                "llm_latency": time.perf_counter() - started_at,
                "llm_retries": float(attempt),
                "llm_status_code": float(response.status_code),
                "llm_fallback_used": 0.0,
            }
            return response.json()

    def get_last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def _extract_content(self, payload: dict) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("Resposta da OpenRouter sem escolhas disponíveis")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("Resposta da OpenRouter sem conteúdo")
        return content

    def _content_to_action(self, content: str) -> AgentAction:
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Resposta do LLM não é JSON válido: {exc}: {content}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError("JSON retornado pelo LLM deve ser um objeto")

        type_name = str(data.get("type", "message")).lower()
        mapping = {
            "message": ActionType.MESSAGE,
            "run_command": ActionType.RUN_COMMAND,
            "apply_patch": ActionType.APPLY_PATCH,
            "write_file": ActionType.WRITE_FILE,
            "read_file": ActionType.READ_FILE,
            "finish": ActionType.FINISH,
        }
        if type_name not in mapping:
            raise RuntimeError(f"Tipo de ação inválido recebido do LLM: {type_name}")

        action_type = mapping[type_name]
        command_value = data.get("command")
        command_list: list[str] | None = None
        if isinstance(command_value, str):
            command_list = command_value.split()
        elif isinstance(command_value, list):
            command_list = [str(part) for part in command_value]

        return AgentAction(
            type=action_type,
            text=data.get("text"),
            command=command_list,
            cwd=data.get("cwd"),
            diff=data.get("diff"),
            path=data.get("path"),
            content=data.get("content"),
        )

    def _ensure_ollama_client(self) -> bool:
        """Garantir que o cliente Ollama esteja inicializado antes do fallback."""
        if self.ollama_client is not None:
            return True
        if self._ollama_unavailable_reason:
            return False
        try:
            from ollama import Client  # type: ignore import-not-found
        except ImportError:  # pragma: no cover - depends on optional dep
            self._ollama_unavailable_reason = (
                "Pacote 'ollama' não encontrado; instale para habilitar fallback."
            )
            return False
        try:
            self.ollama_client = Client()
            return True
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self._ollama_unavailable_reason = (
                f"Não foi possível inicializar o cliente Ollama: {exc}"
            )
            self.ollama_client = None
            return False

    def _send_with_ollama(self, messages: list[dict[str, str]]) -> dict:
        """Send request using Ollama as fallback."""
        started_at = time.perf_counter()
        if self.ollama_client is None:
            raise RuntimeError(
                "Fallback Ollama solicitado sem cliente inicializado."
            )
        try:
            # Format messages for Ollama (ollama expects list of dicts with role and content)
            ollama_messages = [
                {"role": msg["role"], "content": msg["content"]} for msg in messages
            ]
            response = self.ollama_client.chat(
                model="llama3",
                messages=ollama_messages,
                options={"temperature": 0.1},
            )
            content = response["message"]["content"]
            # Format response similar to OpenRouter
            formatted_response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": content,
                        }
                    }
                ]
            }
            latency = time.perf_counter() - started_at
            self._last_metrics.update({
                "ollama_latency": latency,
            })
            return formatted_response
        except Exception as exc:
            self._last_metrics = {
                "llm_latency": time.perf_counter() - started_at,
                "llm_retries": 0.0,
                "llm_status_code": 500.0,  # Internal error
                "llm_fallback_used": 1.0,
                "ollama_error": 1.0,
            }
            raise RuntimeError(f"Falha no fallback Ollama: {exc}") from exc

    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that proposes YAML lists of seeds for A3X optimization based on metrics and history."},
            {"role": "user", "content": prompt},
        ]
        response = self._send_request(messages)
        return self._extract_content(response)


def build_llm_client(llm_config) -> BaseLLMClient:
    llm_type = (llm_config.type or "openrouter").lower()  # Default to openrouter instead of manual
    if llm_type == "manual":
        return ManualLLMClient(llm_config.script)
    if llm_type == "openrouter":
        base_url = llm_config.base_url or llm_config.endpoint
        # Use default model if not specified
        model = llm_config.model or "x-ai/grok-4-fast:free"
        return OpenRouterLLMClient(
            model=model,
            api_key_env=llm_config.api_key_env,
            base_url=base_url,
        )
    raise NotImplementedError(f"LLM type não suportado: {llm_type}")


def _load_actions(path: Path) -> Iterable[AgentAction]:
    if not path.exists():
        raise FileNotFoundError(f"Script manual não encontrado: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    if not isinstance(data, list):
        raise ValueError("Script manual deve ser uma lista de ações")
    for idx, entry in enumerate(data, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Ação inválida na posição {idx}: esperado objeto")
        yield _entry_to_action(entry, idx, path)


def _entry_to_action(entry: dict, idx: int, path: Path) -> AgentAction:
    try:
        type_name = str(entry["type"]).lower()
    except KeyError as exc:  # pragma: no cover - validação
        raise ValueError(f"Ação na posição {idx} em {path} sem campo 'type'") from exc

    mapping = {
        "message": ActionType.MESSAGE,
        "command": ActionType.RUN_COMMAND,
        "patch": ActionType.APPLY_PATCH,
        "write_file": ActionType.WRITE_FILE,
        "read_file": ActionType.READ_FILE,
        "finish": ActionType.FINISH,
    }
    if type_name not in mapping:
        raise ValueError(f"Tipo de ação desconhecido: {type_name} (linha {idx})")

    action_type = mapping[type_name]

    if action_type is ActionType.MESSAGE:
        return AgentAction(type=action_type, text=str(entry.get("text", "")))

    if action_type is ActionType.RUN_COMMAND:
        command = entry.get("command")
        if isinstance(command, str):
            parts = command.split()
        elif isinstance(command, list):
            parts = [str(part) for part in command]
        else:
            raise ValueError(
                f"Ação command inválida na posição {idx}: use lista ou string"
            )
        cwd = entry.get("cwd")
        return AgentAction(type=action_type, command=parts, cwd=cwd)

    if action_type is ActionType.APPLY_PATCH:
        diff = entry.get("diff")
        if not diff:
            raise ValueError(f"Ação patch sem diff na posição {idx}")
        return AgentAction(type=action_type, diff=str(diff))

    if action_type is ActionType.WRITE_FILE:
        path_value = entry.get("path")
        if path_value is None:
            raise ValueError(f"Ação write_file sem path na posição {idx}")
        content = entry.get("content", "")
        return AgentAction(type=action_type, path=str(path_value), content=str(content))

    if action_type is ActionType.READ_FILE:
        path_value = entry.get("path")
        if path_value is None:
            raise ValueError(f"Ação read_file sem path na posição {idx}")
        return AgentAction(type=action_type, path=str(path_value))

    if action_type is ActionType.FINISH:
        return AgentAction(type=action_type, text=str(entry.get("summary", "")))

    raise ValueError(f"Tipo não tratado no script manual: {type_name}")
