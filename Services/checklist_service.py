import io
import json
import logging
from pydantic import BaseModel
from pypdf import PdfReader
from Configs import config
from Services.openai_client import AzureOpenAIClient
from Prompts.prompts import CHECKLIST_PROMPT, GLOBAL_SYSTEM_PROMPT, RELEVANCE_CHECK_PROMPT

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
        self.MIN_TEXT_LENGTH = config.MIN_TEXT_LENGTH
        self.MAX_PDF_SIZE = config.MAX_PDF_SIZE  # 1MB

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
                {"role": "system", "content": GLOBAL_SYSTEM_PROMPT},
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
        if len(text.strip()) < self.MIN_TEXT_LENGTH:
            raise ValueError(
                f"The uploaded PDF contains very little text ({len(text.strip())} characters). "
            )
        chunks = self.split_text(text)
        checklist_items = []
        # Process limited chunks for safety
        for chunk in chunks:
            items = self.generate_checklist_from_chunk(chunk)
            checklist_items.extend(items)

        # Irrelevant PDF check - LLM returned nothing useful
        if not checklist_items:
            logger.warning("No checklist items extracted — document may be irrelevant.")
            raise ValueError(
                "No checklist items found. Please upload a sales training or sales agent guidelines document."
            )

        # fixed deduplication 
        seen = set()
        unique_items = []
        for item in checklist_items:
            label = item.label.strip().lower()
            if label not in seen:
                seen.add(label)
                unique_items.append(item)

        # re-assign clean sequential IDs after dedup
        final = unique_items
        for index, item in enumerate(final):
            item.id = index + 1

        logger.info(f"Generated {len(final)} checklist items")

        # # Remove duplicates while keeping order
        # checklist_items = list(dict.fromkeys(checklist_items))
        # logger.info(f"Generated {len(checklist_items)} checklist items")
        return [item.model_dump() for item in final]
    
    def validate_pdf(self, pdf_bytes, filename: str):
        
        try:
            #For empty files
            if not pdf_bytes:
                return {"is_valid": False, "reason": "File is empty"}

            #For large files
            if len(pdf_bytes) > self.MAX_PDF_SIZE:
                return {"is_valid": False, "reason": "File size exceeds 1 MB limit"}

            #For crashed files
            text = self.extract_text_from_pdf(pdf_bytes)
            if not text or not text.strip():
                return {"is_valid": False, "reason": "No readable content (possibly scanned PDF)"}

            #For short files
            if len(text.strip()) < self.MIN_TEXT_LENGTH:
                return {
                    "is_valid": False,
                    "reason": "Insufficient content (<100 characters)"
                }
            # LLM relevance check (by sending only 2000 characters)
            is_relevant = self._is_relevant_document(text[:2000])

            if not is_relevant:
                return {
                    "is_valid": False,
                    "reason": "Content not relevant"
                }

            return {"is_valid": True,}

        except Exception as e:
            logger.exception(f"Error validating file '{str(filename)}': {str(e)}")
            return {"is_valid": False, "reason": "Corrupted or unreadable PDF"}
        
    def _is_relevant_document(self, text: str) -> bool:
        try:
            sample_text = text
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a strict classifier."},
                    {"role": "user", "content": RELEVANCE_CHECK_PROMPT.format(sample_text=sample_text)}
                ],
                temperature=0
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("Empty response from relevance check, defaulting to relevant")
                return True

            answer = content.strip().upper()
            return answer.startswith("YES") 

        except Exception as e:
            logger.exception(f"Relevance check failed: {str(e)}, defaulting to relevant")
            return True  