from openai import OpenAI

PROMPT = "Please rewrite the following text and return only the rewritten text: "

class Translator():

    def __init__(self, input_language: str, output_language: str):
        self.input_language = input_language
        self.output_language = output_language

    def edit(self, text: str, reference=None) -> str:
        raise NotImplementedError


class GPTBackTranslator(Translator):
    """Translator a text using the GPT model."""

    def __init__(self, input_language: str, output_language: str) -> None:
        """
            Initialize the GPT Translator.
        """

        super().__init__(input_language, output_language)

        self.prompt_detect_language = "Detect the language of the following text and respond with the language name only:\n\n"
        self.prompt_translation = "Translate the following text from {input} to {output} and return only the translated text:\n\n"

        client = OpenAI()
        self.client = client

    def edit(self, text: str, reference=None):
        """Back=Translate the text using the GPT model."""

        # First detect the language
        response = self.client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input= self.prompt_detect_language + text,
        )
        original_language = response.output_text

        translate = self.prompt_translation.format(input=original_language, output=self.output_language)
        response = self.client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input= translate + text,
        )
        response_text = response.output_text

        back_translate = self.prompt_translation.format(input=self.output_language, output=original_language)
        response = self.client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input= back_translate + response_text,
        )
        return response.output_text
