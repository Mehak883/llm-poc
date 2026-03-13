import io
import json
import logging
from pypdf import PdfReader
from Services.openai_client import OpenAIClient
from Prompts.prompts import CHECKLIST_PROMPT

logger = logging.getLogger(__name__)

class ChecklistService:

    def __init__(self):
        self.client = OpenAIClient.get_client()

    def extract_text_from_pdf(self, pdf_bytes):

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logger.info(f"Extracted text length: {len(text)}")
        return text

    def split_text(self, text, chunk_size=4000):

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def generate_checklist_from_chunk(self, text_chunk):

        prompt = CHECKLIST_PROMPT.format(text_chunk=text_chunk)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type":"json_object"},
            temperature=0
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
        result = json.loads(content)

        return result.get("checklist", [])

    def generate_checklist(self, pdf_bytes):

        text = self.extract_text_from_pdf(pdf_bytes)

        if not text.strip():
            raise ValueError("No text extracted from PDF")

        chunks = self.split_text(text)

        checklist_items = []

        # Process limited chunks for safety
        for chunk in chunks[:3]:
            items = self.generate_checklist_from_chunk(chunk)
            checklist_items.extend(items)

        # Remove duplicates while keeping order
        checklist_items = list(dict.fromkeys(checklist_items))

        logger.info(f"Generated {len(checklist_items)} checklist items")

        return checklist_items[:10]