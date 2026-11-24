"""Client minimal de l'API Groq (format compatible OpenAI), avec requests.

La clé vient de la variable d'environnement GROQ_API_KEY (ou du fichier .env à la racine, chargé
par load_env). Elle n'est jamais écrite dans le code, les logs, les messages d'erreur ni le repr.
"""
import os
from pathlib import Path

import requests

API_URL = "https://api.groq.com/openai/v1/chat/completions"
ENV_KEY = "GROQ_API_KEY"
RETRY_STATUS = {500, 502, 503}
ROOT = Path(__file__).resolve().parents[3]


class PlannerError(Exception):
    """Erreur lisible par l'utilisateur (affichée telle quelle dans le chat)."""


class MissingApiKey(PlannerError):
    pass


def load_env() -> None:
    """Charge le .env de la racine du projet s'il existe, sans écraser les variables déjà définies."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)


def has_api_key() -> bool:
    return bool(os.environ.get(ENV_KEY, "").strip())


class GroqClient:
    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.4,
                 max_tokens: int = 2048, timeout_s: float = 30.0, session: requests.Session | None = None):
        key = (api_key or os.environ.get(ENV_KEY, "")).strip()
        if not key:
            raise MissingApiKey(f"No Groq API key: set {ENV_KEY} in the environment or in the .env file.")
        self._api_key = key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._session = session or requests.Session()

    def __repr__(self) -> str:  # jamais la clé
        return f"GroqClient(model={self.model!r})"

    def chat(self, messages: list[dict], json_mode: bool = True) -> str:
        """Envoie la conversation et renvoie le texte de la réponse."""
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = self._post(payload)
        if resp.status_code in RETRY_STATUS:  # Groq surchargé : une seconde chance
            resp = self._post(payload)

        if resp.status_code == 401:
            raise PlannerError(f"Groq rejected the API key. Check {ENV_KEY}.")
        if resp.status_code == 429:
            raise PlannerError("Groq rate limit reached. Wait a minute and try again.")
        if resp.status_code >= 400:
            raise PlannerError(f"Groq returned an error ({resp.status_code}): {_error_message(resp)}")
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise PlannerError("Groq sent an answer in an unexpected format.") from None

    def _post(self, payload: dict) -> requests.Response:
        """POST avec une nouvelle tentative si la connexion tombe (coupure réseau, connexion
        réutilisée fermée par le serveur). Le type d'erreur est donné, jamais la clé."""
        for attempt in range(2):
            try:
                return self._session.post(API_URL, json=payload, timeout=self.timeout_s,
                                          headers={"Authorization": f"Bearer {self._api_key}",
                                                   "Content-Type": "application/json"})
            except requests.Timeout:
                raise PlannerError("Groq took too long to answer. Try again in a moment.") from None
            except requests.RequestException as e:
                if attempt == 0 and isinstance(e, requests.ConnectionError):
                    continue
                raise PlannerError(f"Could not reach Groq ({type(e).__name__}). "
                                   "Check your internet connection and try again.") from None
        raise AssertionError("unreachable")


def _error_message(resp: requests.Response) -> str:
    """Message d'erreur de Groq ({"error": {"message": ...}}), raccourci."""
    try:
        msg = resp.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        msg = "no details"
    return str(msg)[:200]
