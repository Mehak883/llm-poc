#  This file will have all the prompts which are used by LLM for different endpoints.

CHECKLIST_PROMPT="""You are a sales training expert.
From the following training document extract checklist items
that a sales agent must follow during a conversation.
Rules:
- Return 6 to 10 checklist items
- Each item must be short and actionable
- Avoid duplicates
- Return JSON only in this format:

{{
 "checklist": [
   "item 1",
   "item 2"
 ]
}}

Document:
{text_chunk}"""  

INTENT_PROMPT="""You are an expert and strict evaluator analyzing a phone call between a customer and a sales agent.
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

CUSTOMER_SATISFACTION_PROMPT=""" You are a customer satisfaction evaluator.
    Read the following last messages between a customer with role agent and a sales agent with role user.

    Conversation:
    {formatted_conversation}

    Based on the customer's tone, mood, and the final resolution,
    rate their satisfaction on a scale from 0 to 10.

    0 = Extremely Dissatisfied
    5 = Neutral
    10 = Extremely Satisfied

    Respond with ONLY the numeric score (no explanation, no text)."""

ASSIST_PROMPT="""You are a real-time sales coach.
                        Conversation:
                        {formatted_conversation}
                        Checklist items the sales agent must still cover:
                        {checklist_prompt}
                        Return ALL listed checklist items using this format:

                        1 done
                        2 pending
                        3 active"""