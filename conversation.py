from groq import Groq
import json

SYSTEM_PROMPT = """You are a compassionate mental health screening assistant 
called LUMINA-D AI. Your role is to:

1. Have a warm, empathetic conversation to understand how the user is feeling
2. Gently ask about key depression indicators:
   - Sleep patterns
   - Energy levels
   - Interest in activities
   - Appetite changes
   - Concentration
   - Feelings of worthlessness
   - Social withdrawal
3. Extract structured information from the conversation
4. Never diagnose - always recommend professional help
5. Be supportive and non-judgmental
6. Keep responses concise (2-3 sentences max)
7. Ask ONE question at a time

When you have gathered enough information (at least 5-6 exchanges),
end with: [READY_FOR_ANALYSIS]

Important: You are a screening tool only, not a replacement for 
professional mental health care."""

EXTRACTION_PROMPT = """Based on this conversation, extract depression 
indicators as JSON. Return ONLY valid JSON, nothing else:
{
    "sleep_hours": <number 1-12 or null>,
    "energy_level": <1-5 scale or null>,
    "interest_loss": <true/false or null>,
    "appetite_change": <"increased"/"decreased"/"normal" or null>,
    "concentration": <1-5 scale or null>,
    "worthlessness": <true/false or null>,
    "social_withdrawal": <true/false or null>,
    "summary": "<brief 1-sentence summary of key concerns>"
}"""

EXPLANATION_PROMPT = """You are a compassionate mental health assistant.
Explain these depression screening results to the user in a warm,
supportive way. Be honest but gentle. Always recommend professional help.
Keep it to 3-4 sentences.

Results:
- Risk Level: {risk_level}
- Confidence: {confidence:.1%}
- Modalities analyzed: {modalities}
- Key finding: {key_finding}"""


class ConversationManager:
    def __init__(self, groq_api_key: str):
        self.client  = Groq(api_key=groq_api_key)
        self.history = []
        self.model   = "llama-3.3-70b-versatile"

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        response = self.client.chat.completions.create(
            model      = self.model,
            messages   = messages,
            max_tokens = 200,
            temperature= 0.7
        )

        assistant_msg = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_msg})
        return assistant_msg

    def is_ready_for_analysis(self) -> bool:
        if len(self.history) < 10:
            return False
        return "[READY_FOR_ANALYSIS]" in self.history[-1]["content"]

    def extract_features(self) -> dict:
        conversation_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in self.history
        ])
        response = self.client.chat.completions.create(
            model      = self.model,
            messages   = [
                {"role": "system",  "content": "Extract information as JSON only."},
                {"role": "user",    "content": f"{EXTRACTION_PROMPT}\n\nConversation:\n{conversation_text}"}
            ],
            max_tokens = 300,
            temperature= 0.1
        )
        try:
            text = response.choices[0].message.content.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except:
            return {}

    def explain_results(self, risk_level, confidence, modalities, key_finding) -> str:
        prompt = EXPLANATION_PROMPT.format(
            risk_level  = risk_level,
            confidence  = confidence,
            modalities  = ", ".join(modalities),
            key_finding = key_finding
        )
        response = self.client.chat.completions.create(
            model      = self.model,
            messages   = [
                {"role": "system", "content": "You are a compassionate mental health assistant."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens = 200,
            temperature= 0.7
        )
        return response.choices[0].message.content

    def reset(self):
        self.history = []
