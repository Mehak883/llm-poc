import logging
from Services.openai_client import OpenAIClient
from Services.session_manager import SessionManager

logger = logging.getLogger(__name__)


class AssistService:

    def __init__(self):

        self.client = OpenAIClient.get_client()
        self.session_manager = SessionManager()

    def format_transcript(self, transcript):

        recent = transcript[-12:]
        lines = []
        for t in recent:
            role = t.get("role", "")
            msg = t.get("message", "")
            lines.append(f"{role}: {msg}")

        return "\n".join(lines)

    def checklist_to_prompt(self, checklist):

        lines = ["checklist:"]
        for item in checklist:
            lines.append(f"{item['id']} {item['label']}")
        return "\n".join(lines)

    def parse_output(self, text):

        checklist = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                parts = line.split()
                if len(parts) >= 2:
                    item_id = int(parts[0].replace(":", ""))
                    checklist.append({
                        "id": item_id,
                        "status": parts[1]
                    })
        return checklist

    def analyze(self, conv_id, transcript, checklist=None):

        try:
            session = self.session_manager.get_session(conv_id)
            if session is None:
                if not checklist:
                    return {
                        "checklist": [],
                        "error": "Session not found and checklist missing"
                    }

                self.session_manager.create_session(conv_id, checklist)
            items_to_check = self.session_manager.get_pending_items(conv_id)
            if not items_to_check:
                return {
                    "checklist": self.session_manager.build_response(conv_id)
                }
            formatted_conversation = self.format_transcript(transcript)
            checklist_prompt = self.checklist_to_prompt(items_to_check)
            prompt = f"""
                        You are a real-time sales coach.
                        Conversation:
                        {formatted_conversation}
                        Checklist items the sales agent must still cover:
                        {checklist_prompt}
                        Return ALL listed checklist items using this format:

                        1 done
                        2 pending
                        3 active
                        """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only formatted checklist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=120
            )

            output = response.choices[0].message.content or ""
            parsed = self.parse_output(output)
            self.session_manager.update_state(conv_id, parsed)
            return {
                "checklist": self.session_manager.build_response(conv_id)
            }

        except Exception as e:
            logger.exception("AssistService error")
            return {
                "error": str(e)
            }

    def end_session(self, conv_id):
        self.session_manager.end_session(conv_id)