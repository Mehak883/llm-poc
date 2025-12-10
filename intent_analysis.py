import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculate_words_spoken(transcript):
    return sum(len(t.get("message", "").split()) for t in transcript if t.get("message"))

def analyze_customer_satisfaction(transcript):
    """
    Analyze customer satisfaction using an LLM on the last messages.
    Returns only the satisfaction score (0-10).
    """

    if not transcript:
        return 0.0

    # Take last 10 messages or fewer
    last_messages = transcript[-10:]

    # Combine into readable dialogue text
    formatted_conversation = "\n".join([
        f"{t.get('role', '').capitalize()}: {t.get('message', '')}"
        for t in last_messages if t.get("message")
    ])

    if not formatted_conversation.strip():
        return 0.0

    # LLM prompt
    prompt = f"""
    You are a customer satisfaction evaluator.
    Read the following last messages between a customer with role agent and a sales agent with role user.

    Conversation:
    {formatted_conversation}

    Based on the customer's tone, mood, and the final resolution,
    rate their satisfaction on a scale from 0 to 10.

    0 = Extremely Dissatisfied
    5 = Neutral
    10 = Extremely Satisfied

    Respond with ONLY the numeric score (no explanation, no text).
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only the numeric score (0-10)."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        content = (res.choices[0].message.content or "").strip()


        # Try to extract a float from response
        try:
            score = float(content)
            return max(0.0, min(10.0, round(score, 1)))
        except ValueError:
            # If model adds extra text accidentally
            import re
            match = re.search(r"(\d+(\.\d+)?)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(10.0, round(score, 1)))
            else:
                return 0.0

    except Exception as e:
        print(f"Error in satisfaction analysis: {e}")
        return 0.0

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

    In addition, identify 3 fixed key moments and rate each:

    Key Moments:
    1. Opening Response
    2. Problem Investigation
    3. Resolution Offer

    For each moment:
    - "moment_title" → one of the above
    - "level" → one of ["Excellent", "Very Good", "Good", "Moderate", "Needs Improvement"]
    - "moment_feedback" → short descriptive reason or example quote

    Example Output:
    [
      {{"moment_title": "Opening Response", "level": "Excellent", "moment_feedback": "Perfect empathy opening."}},
      {{"moment_title": "Problem Investigation", "level": "Good", "moment_feedback": "Identified issue quickly but lacked depth."}},
      {{"moment_title": "Resolution Offer", "level": "Excellent", "moment_feedback": "Clear resolution and follow-up commitment."}}
    ]

    Also identify the exact customer or agent sentence that best represents the "Opening Response" moment. Return it as "opening_response_sentence".
    Return the final output strictly adhering to the following JSON schema:

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
    result["words_spoken"] = calculate_words_spoken(transcript)
    result["conversation_id"] = conversation_id
    result["customer_satisfaction_score"] = analyze_customer_satisfaction(transcript)
    return result
