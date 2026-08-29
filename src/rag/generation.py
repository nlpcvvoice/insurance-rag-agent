from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    answer: str
    sources: List[str]
    model: str


class LLMGenerator:
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        self._client = None

    def _get_client(self, system_instruction: Optional[str] = None):
        if self._client is None or system_instruction is not None:
            try:
                from vertexai.generative_models import GenerativeModel
                import vertexai
                vertexai.init()
                self._client = GenerativeModel(
                    self.model,
                    system_instruction=system_instruction,
                )
            except ImportError:
                raise ImportError("vertexai not installed. Run: pip install google-cloud-aiplatform")
        return self._client

    def generate(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        if system_prompt is None:
            system_prompt = """You are an insurance knowledge assistant.
Answer questions based ONLY on the provided context below.
- If the context is EMPTY or does NOT contain enough information to answer,
  say so clearly. Do NOT use outside knowledge or guess.
- Base your answer strictly on the given context.
- Cite your sources when possible."""

        context_text = "\n\n".join([f"Source {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

        client = self._get_client(system_prompt)
        response = client.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": 1024,
            },
        )

        return GenerationResult(
            answer=response.text,
            sources=[f"Source {i+1}" for i in range(len(context))],
            model=self.model,
        )

    def generate_with_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        client = self._get_client()
        response = client.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": 1024,
            },
        )
        return response.text
