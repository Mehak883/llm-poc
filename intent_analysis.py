import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_call_structured(conversation_id, transcript):

    
    # Only user messages for intent detection
    user_messages = [
        t.get("message", "") for t in transcript if t.get("role") == "user"
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

    prompt = f"""
    You are an expert and strict evaluator analyzing a phone call between a customer and a sales agent.
    You will always be analyzing the scores of the sales agent. Give the true scores.

    conversation_id MUST always be: {conversation_id}

    Provide structured JSON ONLY (strict mode) following the schema.

    TRANSCRIPT:
    {json.dumps(user_messages, indent=2)}

    Title must ALWAYS be:
    "Sales Agent Performance Review"

    Scoring rules:
    - empathy, problem_solving, communication_clarity, product_knowledge, call_efficiency → 10 to 100
    - "what_you_did_well" : minimum 4 bullet points
    - "areas_of_improvement" : minimum 4 bullet points
    - "intent" : short phrase (e.g. "loan enquiry", "complaint", "account issue")
    """

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
                        "empathy": {"type": "number"},
                        "problem_solving": {"type": "number"},
                        "communication_clarity": {"type": "number"},
                        "product_knowledge": {"type": "number"},
                        "call_efficiency": {"type": "number"}
                    },
                    "required": [
                        "empathy",
                        "problem_solving",
                        "communication_clarity",
                        "product_knowledge",
                        "call_efficiency"
                    ]
                },

            },
            "required": [
                "conversation_id",
                "intent",
                "feedback",
                "performance_scores",
            ]
        },
        "strict": True
    }

    res = client.chat.completions.create(
        model="gpt-4o-mini",
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
        return {"error": "Model returned no content"}

    result = json.loads(content)

    result["conversation_id"] = conversation_id
    return result
