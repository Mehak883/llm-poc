import time
import logging

logger = logging.getLogger(__name__)

class SessionManager:

    SESSION_TIMEOUT_SECONDS = 30 * 60

    def __init__(self):
        logger.info("SessionManager initialized")
        self.sessions = {}

    def normalize_checklist(self, checklist):
        if not checklist:
            return []
        if isinstance(checklist[0], str):
            return [{"id": i + 1, "label": item} for i, item in enumerate(checklist)]
        logger.debug("Normalizing checklist format")
        return checklist

    def create_session(self, conv_id, checklist):

        normalized = self.normalize_checklist(checklist)
        self.sessions[conv_id] = {
            "checklist": normalized,
            "state": {item["id"]: "pending" for item in normalized},
            "last_updated": time.time()
        }
        logger.info(f"Session created for {conv_id}")

    def get_session(self, conv_id):

        logger.debug(f"Fetching session for conversation: {conv_id}")
        session = self.sessions.get(conv_id)
        if not session:
            return None
        if time.time() - session["last_updated"] > self.SESSION_TIMEOUT_SECONDS:
            del self.sessions[conv_id]
            return None
        
        return session

    def update_state(self, conv_id, parsed):

        session = self.sessions.get(conv_id)
        if not session:
            return
        for item in parsed:
            item_id = item["id"]
            new_status = item["status"]
            current = session["state"].get(item_id)
            if current == "done":
                continue
            session["state"][item_id] = new_status
        session["last_updated"] = time.time()

    def get_pending_items(self, conv_id):

        session = self.sessions[conv_id]
        logger.debug(f"Fetching pending checklist items for {conv_id}")
        return [
            item for item in session["checklist"]
            if session["state"][item["id"]] in ("pending", "active")
        ]

    def build_response(self, conv_id):

        session = self.sessions[conv_id]
        logger.debug(f"Fetching session for conversation: {conv_id}")
        return [
            {"id": item["id"], "status": session["state"][item["id"]]}
            for item in session["checklist"]
        ]

    def end_session(self, conv_id):

        if conv_id in self.sessions:
            del self.sessions[conv_id]
            logger.info(f"Session ended {conv_id}")