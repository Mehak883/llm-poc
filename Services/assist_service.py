import logging
from typing import Literal
from pydantic import BaseModel
from Services.openai_client import AzureOpenAIClient
from Services.session_manager import SessionManager
from Prompts.prompts import ASSIST_PROMPT, GLOBAL_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Pydantic models for strict output
class AssistItem(BaseModel):
    id: int
    status: Literal["pending", "done"]

class AssistResponse(BaseModel):
    checklist: list[AssistItem]


class AssistService:

    def __init__(self):

        self.client = AzureOpenAIClient.get_client()
        self.session_manager = SessionManager()

    def format_transcript(self, transcript):

        recent = transcript[-12:]
        lines = []
        for t in recent:
            role = t.role
            msg = t.message
            lines.append(f"{role}: {msg}")

        return "\n".join(lines)

    def checklist_to_prompt(self, checklist):

        lines = ["checklist:"]
        for item in checklist:
            lines.append(f"{item['id']} {item['label']}")
        return "\n".join(lines)

    # def parse_output(self, text):

    #     checklist = []
    #     for line in text.split("\n"):
    #         line = line.strip()
    #         if not line:
    #             continue
    #         if line[0].isdigit():
    #             parts = line.split()
    #             if len(parts) >= 2:
    #                 item_id = int(parts[0].replace(":", ""))
    #                 checklist.append({
    #                     "id": item_id,
    #                     "status": parts[1]
    #                 })
    #     return checklist

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
            session = self.session_manager.get_session(conv_id)
            if session is None:
                logger.error(f"Session unavailable for {conv_id} after create attempt")
                return {"error": "Session could not be established"}
 
            items_to_check = self.session_manager.get_pending_items(conv_id)
            if not items_to_check:
                return {
                    "checklist": self.session_manager.build_response(conv_id)
                }
            logger.info(f"the transcript is {transcript}")
            formatted_conversation = self.format_transcript(transcript)
            checklist_prompt = self.checklist_to_prompt(items_to_check)
            prompt = ASSIST_PROMPT.format(formatted_conversation=formatted_conversation, checklist_prompt=checklist_prompt)

            response = self.client.chat.completions.parse(
                model="gpt-4o-mini",
                response_format=AssistResponse,
                messages=[
                    {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=200
            )

            output = response.choices[0].message

            if output.refusal:                                  # handle refusal
                logger.warning(f"LLM refused: {output.refusal}")
                return {"error": "LLM refused to respond"}

            if output.parsed is None:
                logger.error(f"Parsing failed. Raw content: {output.content}")
                return []
            parsed = output.parsed
            self.session_manager.update_state(
                conv_id,
                [item.model_dump() for item in parsed.checklist]
            )
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