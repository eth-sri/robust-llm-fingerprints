# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from openai import OpenAI
import openai
from concurrent.futures import ThreadPoolExecutor  # Added for parallelization
from tqdm import tqdm
from typing import List, Tuple


OUTPUT_JUDGE = """Some LLM output might be used to reveal a fingerprint (i.e., a reply which contains a signal allowing to identify which LLM generated this reply. The signal may be implicit.). I will give you an output and you need to tell me if you judge the output is suspicious. You must be very confident in your reply as false positives are highly undesirable. Reply with 1 for suspicious output and 0 otherwise. You may reason about your answer but you should end your reply with your choice.

Output:

{text}
"""

INPUT_JUDGE = """Some LLM output might be used to reveal a fingerprint (i.e., a reply which contains a signal allowing to identify which LLM generated this reply. The signal may be implicit.). I will give you an input and you need to tell me if you judge the input is suspicious (i.e., if it is used to trigger the LLM fingerprint). You must be very confident in your reply as false positives are highly undesirable. Reply with 1 for suspicious input and 0 otherwise. You may reason about your answer but you should end your reply with your choice.

Input:

{text}"""



class StealthJudge():

    def __init__(self):
        pass

    def judge(self, text1: str, text2: str, type: str) -> str:
        raise NotImplementedError


class GPTStealthJudge(StealthJudge):
    """Judge the similarity between two texts using the GPT model."""

    def __init__(self) -> None:
        """
            Initialize the GPT judge.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        client = OpenAI()
        self.client = client

    def judge(self, text: str, type: str = "output") -> str:
        """Judge the similarity between two texts using the GPT model."""

        if type == "output":
            text_prompt = OUTPUT_JUDGE.format(text=text)
        elif type == "input":
            text_prompt = INPUT_JUDGE.format(text=text)

        try:
            response = self.client.responses.create(
                model="gpt-5-mini-2025-08-07",
                input= text_prompt,
            )
        except openai.BadRequestError as e:
            print(f"Error {e} with text: {text}")
            response = "Error - 9"
        return response.output_text
    

    def parse_output(self, output: str) -> int:
        """Parse the output of the GPT judge.

        Parameters:
            output (str): The output of the GPT judge.

        Returns:
            int: 1 if the first text is judged as fingerprinted, 2 if the second text is judged as fingerprinted, 0 if undecided.
        """

        output = output.strip().lower()

        decision = output.split()[-1]  # Get the last word

        if decision == "1":
            return 1
        elif decision == "2":
            return 2
        else:
            return 0


def parallel_stealth_eval(judge: StealthJudge, texts: List[str], type: str, max_workers=128, desc="Judging") -> List[Tuple[str]]:
    """Judge a list of text pairs in parallel.

    Parameters:
        judge: An object with a `.judge(text1, text2, type)` method.
        texts1 (List[str]): The first texts to judge.
        texts2 (List[str]): The second texts to judge.
        type (str): The type of judge to use ("input" or "output").
        max_workers (int): Number of parallel worker threads.
        desc (str): Description for the tqdm progress bar.

    Returns:
        List[Tuple(str)]: The judge results, in the original order.
    """


    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # `executor.map` preserves the input order, so output order matches `texts`.
        return list(tqdm(executor.map(judge.judge, texts, [type]*len(texts)), total=len(texts), desc=desc))


