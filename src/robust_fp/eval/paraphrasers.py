from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor  # Added for parallelization

PARAPHRASE_PROMPT = "Please rewrite the following text and return only the rewritten text: "
OUTPUT_PARAPHRASE_PROMPT = PARAPHRASE_PROMPT


def parallel_paraphrase(paraphraser, texts, max_workers=128, desc="Paraphrasing"):
    """Paraphrase a list of texts in parallel.

    Parameters:
        paraphraser: An object with an `.edit(text)` method.
        texts (List[str]): The texts to paraphrase.
        max_workers (int): Number of parallel worker threads.
        desc (str): Description for the tqdm progress bar.

    Returns:
        List[str]: The paraphrased texts, in the original order.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # `executor.map` preserves the input order, so output order matches `texts`.
        return list(tqdm(executor.map(paraphraser.edit, texts), total=len(texts), desc=desc))

class Paraphraser():

    def __init__(self):
        pass

    def edit(self, text: str, reference=None) -> str:
        raise NotImplementedError


class GPTParaphraser(Paraphraser):
    """Paraphrase a text using the GPT model."""

    def __init__(self, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.prompt = prompt

        client = OpenAI()
        self.client = client

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""

        response = self.client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input= self.prompt + text,
        )
        return response.output_text
