"""Lectures / écritures utilisées par l'app et le rapport de séance."""


class SessionRepository:
    def start_session(self, user_id: int) -> int:
        raise NotImplementedError

    def save_rep(self, set_id: int, rep_summary: dict) -> int:
        raise NotImplementedError

    def log_event(self, rep_id: int, fault_code: str, tip: str) -> None:
        raise NotImplementedError

    def session_report(self, session_id: int) -> dict:
        raise NotImplementedError
