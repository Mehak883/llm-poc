import json
import logging
from Services.openai_client import AzureOpenAIClient
from Configs import config
from Prompts.prompts import INTENT_PROMPT, CUSTOMER_SATISFACTION_PROMPT

logger = logging.getLogger(__name__)

class IntentAnalysis:

    def __init__(self):
        self.client = AzureOpenAIClient.get_client()
        logger.info("OpenAI client initialized")

    def calculate_words_spoken(self, transcript):
        word_count = sum(len(t.message.split()) for t in transcript if t.message)
        logger.debug(f"Calculated total words spoken: {word_count}")
        return word_count

    def analyze_customer_satisfaction(self, transcript):
        """
        Analyze customer satisfaction using an LLM on the last messages.
        Returns only the satisfaction score (0-10).
        """

        if not transcript:
            logger.warning("Transcription is empty. Returning customer satisfaction score of 0.0.")
            return 0.0

        # Take last 10 messages or fewer
        last_messages = transcript[-10:]

        # Combine into readable dialogue text
        formatted_conversation = "\n".join([
            f"{t.role.capitalize()}: {t.message}"
            for t in last_messages if t.message
        ])

        if not formatted_conversation.strip():
            logger.warning("No valid messages found in the last 10 entries. Returning customer satisfaction score of 0.0.")
            return 0.0

        # LLM prompt
        prompt = CUSTOMER_SATISFACTION_PROMPT.format(formatted_conversation=formatted_conversation)

        try:
            res = self.client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "customer_satisfaction_schema",
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 10
                                }
                            },
                            "required": ["score"]
                        },
                        "strict": True
                    }
                },
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            content = res.choices[0].message.content
            if not content:
                logger.error("Model returned no content")
                return None
            result = json.loads(content)

            score = round(float(result["score"]), 1)
            logger.info(f"Customer satisfaction score: {score}")

            return score
        
        except Exception as e:
            logger.exception(f"Error in satisfaction analysis: {e}")
            return 0.0

    def analyze_call_structured(self, conversation_id, transcript):
        # Only user messages for intent detection
        user_messages = [
            t.message for t in transcript if t.role == "user"
        ]

        if not user_messages or all(m.strip() == "" for m in user_messages):
            return {
                "conversation_id": conversation_id,
                "intent": "No valid user conversation",
                "feedback": {
                    "title": "Insufficient Data",
                    "what_you_did_well": [],
                    "areas_of_improvement": ["No user messages found in this conversation."]
                },
                "performance_scores": {
                    "empathy": 0,
                    "problem_solving": 0,
                    "communication_clarity": 0,
                    "product_knowledge": 0,
                    "call_efficiency": 0
                }
            }
        
        user_transcript = json.dumps(user_messages, indent=2)

        prompt = INTENT_PROMPT.format(conversation_id=conversation_id, user_messages=user_transcript)

        schema = {
            "name": "sales_agent_call_review",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "conversation_id": {"type": "string"},
                    "intent": {"type": "string"},

                    "feedback": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "what_you_did_well": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "areas_of_improvement": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": [
                            "title",
                            "what_you_did_well",
                            "areas_of_improvement"
                        ]
                    },

                    "performance_scores": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "empathy": {"type": "number", "minimum": 10, "maximum": 100},
                            "problem_solving": {"type": "number", "minimum": 10, "maximum": 100},
                            "communication_clarity": {"type": "number", "minimum": 10, "maximum": 100},
                            "product_knowledge": {"type": "number", "minimum": 10, "maximum": 100},
                            "call_efficiency": {"type": "number", "minimum": 10, "maximum": 100}
                        },
                        "required": [
                            "empathy",
                            "problem_solving",
                            "communication_clarity",
                            "product_knowledge",
                            "call_efficiency"
                        ]
                    },

                    "key_moments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "moment_title": {
                                    "type": "string",
                                    "enum": ["Opening Response", "Problem Investigation", "Resolution Offer"]
                                },
                                "level": {
                                    "type": "string",
                                    "enum": ["Excellent", "Very Good", "Good", "Moderate", "Needs Improvement"]
                                },
                                "moment_feedback": {"type": "string"}
                            },
                            "required": ["moment_title", "level", "moment_feedback"]
                        },
                        "additionalProperties": False,
                        "minItems": 3,
                        "maxItems": 3
                    },
                    "opening_response_sentence": {"type": "string"}


                },
                "required": [
                    "conversation_id",
                    "intent",
                    "feedback",
                    "performance_scores",
                    "key_moments",
                    "opening_response_sentence"
                ]
            },
            "strict": True
        }

        try:
            res = self.client.chat.completions.create(
                model=config.AZURE_OPENAI_DEPLOYMENT_NAME,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sales_schema",
                        "schema": schema["schema"],      
                        "strict": True
                    }
                },
                messages=[
                    {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = res.choices[0].message.content

            if content is None:
                logger.error(f"Model returned no content for conversation_id: {conversation_id}")
                return {"error": "Model returned no content"}

            result = json.loads(content)
            result["words_spoken"] = self.calculate_words_spoken(transcript)
            result["conversation_id"] = conversation_id
            result["customer_satisfaction_score"] = self.analyze_customer_satisfaction(transcript)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error while analyzing feedback for conversation_id {conversation_id}: {e}", exc_info=True)
            return {"error": f"JSON parsing error: {e}"}
        except Exception as e:
            logger.exception(f"Error in structured call analysis for conversation_id {conversation_id}: {e}")
            return {"error": str(e)}
