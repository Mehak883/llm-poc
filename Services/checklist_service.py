import io
import json
import logging
import logging
from pydantic import BaseModel
from pypdf import PdfReader
from Services.openai_client import AzureOpenAIClient
from Prompts.prompts import CHECKLIST_PROMPT

logger = logging.getLogger(__name__)

# Pydantic models for strict output
class ChecklistItem(BaseModel):
    id: int
    label: str

class ChecklistResponse(BaseModel):
    checklist: list[ChecklistItem]

class ChecklistService:

    def __init__(self):
        self.client = AzureOpenAIClient.get_client()

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
        response = self.client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format=ChecklistResponse,
            temperature=0
        )
        message = response.choices[0].message

        if message.refusal:       # handle refusal
            logger.warning(f"LLM refused: {message.refusal}")
            return []

           
        if message.parsed is None:
            logger.error(f"Parsing failed. Raw content: {message.content}")
            return []
        return message.parsed.checklist    

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

        # fixed deduplication 
        seen = set()
        unique_items = []
        for item in checklist_items:
            label = item.label.strip().lower()
            if label not in seen:
                seen.add(label)
                unique_items.append(item)

        # re-assign clean sequential IDs after dedup
        final = unique_items[:10]
        for index, item in enumerate(final):
            item.id = index + 1

        logger.info(f"Generated {len(final)} checklist items")

        # # Remove duplicates while keeping order
        # checklist_items = list(dict.fromkeys(checklist_items))
        # logger.info(f"Generated {len(checklist_items)} checklist items")
        return [item.model_dump() for item in final]