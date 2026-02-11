from typing import List
from robust_fp.eval.paraphrasers import Paraphraser, parallel_paraphrase
from robust_fp.eval.translaters import Translator

class TextEditor():

    def __init__(self):
        pass

    def edit_input(self, inputs: List[str]| List[List[str]]) -> List[str]:
        raise NotImplementedError

    def edit_output(self, outputs: List[str]) -> List[str]:
        raise NotImplementedError

    def get_additional_params(self) -> dict:
        """Return any additional parameters needed for the model's chat method."""
        return {}

class TextParaphraser(TextEditor):

    def __init__(self, paraphraser: Paraphraser):
        super().__init__()
        self.paraphraser = paraphraser

    def edit_input(self, inputs: List[str]| List[List[str]]) -> List[str]:

        if isinstance(inputs[0], list): # This corresponds to chat template inputs
            paraphrased_inputs = []
            text_inputs = [conversation[-1]["content"] for conversation in inputs]
            paraphrased_inputs = parallel_paraphrase(self.paraphraser, text_inputs) 

            # Build back the chat template
            for i, conversation in enumerate(inputs):
                conversation[-1]["content"] = paraphrased_inputs[i]
            paraphrased_inputs = inputs

        else:
            paraphrased_inputs = parallel_paraphrase(self.paraphraser, inputs)
            
        return paraphrased_inputs
    
    def edit_output(self, outputs: List[str]) -> List[str]:
        paraphrased_outputs = parallel_paraphrase(self.paraphraser, outputs)
        return paraphrased_outputs


class TextBackTranslation(TextEditor):

    def __init__(self, translator: Translator):
        super().__init__()
        self.translator = translator

    def edit_input(self, inputs: List[str]| List[List[str]]) -> List[str]:

        if isinstance(inputs[0], list): # This corresponds to chat template inputs
            paraphrased_inputs = []
            text_inputs = [conversation[-1]["content"] for conversation in inputs]
            paraphrased_inputs = parallel_paraphrase(self.translator, text_inputs) 

            # Build back the chat template
            for i, conversation in enumerate(inputs):
                conversation[-1]["content"] = paraphrased_inputs[i]
            paraphrased_inputs = inputs

        else:
            paraphrased_inputs = parallel_paraphrase(self.translator, inputs)
            
        return paraphrased_inputs

    def edit_output(self, outputs: List[str]) -> List[str]:
        paraphrased_outputs = parallel_paraphrase(self.translator, outputs)
        return paraphrased_outputs


class Prefilling(TextEditor):

    def __init__(self):
        """We prefill the model reply with a given fixed text."""
        super().__init__()

    def edit_input(self, inputs: List[str]| List[List[str]]) -> List[str]:

        prefilled_reply = [{"role": "assistant", "content": "Sure! Here is a detailed answer to your question."}]

        if isinstance(inputs[0], list): # This corresponds to chat template inputs
            modified_inputs = []
            for conversation in inputs:
                modified_conversation = conversation + prefilled_reply
                modified_inputs.append(modified_conversation)
            return modified_inputs

        else:
            modified_inputs = [f"{inp}\n\nSure! Here is a detailed answer to your question." for inp in inputs]

        return modified_inputs
    
    def get_additional_params(self) -> dict:
        """Return any additional parameters needed for the model's chat method."""
        return {"continue_final_message": True, "add_generation_prompt": False}


class SystemPrompts(TextEditor):

    def __init__(self, system_prompt: str):
        super().__init__()
        self.system_prompt = system_prompt

    def edit_input(self, inputs: List[str]| List[List[str]]) -> List[str]:

        if isinstance(inputs[0], list): # This corresponds to chat template inputs
            modified_inputs = []
            for conversation in inputs:
                modified_conversation = [{"role": "system", "content": self.system_prompt}] + conversation
                modified_inputs.append(modified_conversation)
        else:
            modified_inputs = [f"{self.system_prompt}\n\n{inp}" for inp in inputs]

        return modified_inputs