# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

# Forked from: https://github.com/THU-BPM/MarkLLM
# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import nltk
import torch
import random
from nltk.corpus import wordnet
from translate import Translator
from nltk.tokenize import sent_tokenize
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM, AutoTokenizer
from pydantic import BaseModel
from strenum import StrEnum
from typing import List, Tuple
import openai
import time    
import numpy as np

class TextEditorType(StrEnum):
    
    RANDOM_WALK_ATTACK = "random_walk_attack"
    GPT_PARAPHRASER = "gpt_paraphraser"
    DIPPER_PARAPHRASER = "dipper_paraphraser"
    WORD_DELETION = "word_deletion"
    SYNONYM_SUBSTITUTION = "synonym_substitution"
    CONTEXT_AWARE_SYNONYM_SUBSTITUTION = "context_aware_synonym_substitution"
    TOKEN_SUBSTITUTION = "token_substitution"
    TOKEN_INSERTION = "token_insertion"
    TOKEN_DELETION = "token_deletion"

    def get_text_editor(self, config: dict) -> "TextEditor":

        if self.value == "gpt_paraphraser":
            return GPTParaphraser(**config)
        elif self.value == "dipper_paraphraser":
            return DipperParaphraser(**config)
        elif self.value == "word_deletion":
            return WordDeletion(**config)
        elif self.value == "synonym_substitution":
            return SynonymSubstitution(**config)
        elif self.value == "context_aware_synonym_substitution":
            return ContextAwareSynonymSubstitution(**config)
        elif self.value == "token_substitution":
            return TokenEditor(**config)
        elif self.value == "token_insertion":
            return TokenEditor(**config)
        elif self.value == "token_deletion":
            return TokenEditor(**config)
        else:
            raise ValueError(f"Text editor type '{self.value}' not supported.")
    
    def short_str(self, config: dict) -> str:
        
        name = self.value
        
        if self.value in ["word_deletion", "token_substitution", "token_insertion", "token_deletion"]:
            name += f"_{config['ratio']:0.2f}"
        
        return name

class TextEditorConfiguration(BaseModel):
    text_editor_types: List[TextEditorType]
    text_editor_params: List[dict] = []

    def get_items(self) -> Tuple[List[TextEditorType], List[dict]]:
        return self.text_editor_types, self.text_editor_params


class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None) -> str:
        return text
    
class TokenEditor(TextEditor):
    
    def __init__(self, tokenizer_name: str, ratio: float, type: str):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.ratio = ratio
        self.type = type
        self.vocab_size = self.tokenizer.vocab_size
        
    def edit(self, text, reference=None):
        # Tokenize and convert to numpy array
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = np.array(tokens)
        num_tokens = len(tokens)
        
        # Determine number of tokens to edit
        n_tokens_to_edit = int(self.ratio * num_tokens)
        
        if self.type == "substitution":
            selected_tokens_indices = np.random.choice(num_tokens, n_tokens_to_edit, replace=False)
            substitute_tokens = np.random.randint(0, self.vocab_size, n_tokens_to_edit)
            tokens[selected_tokens_indices] = substitute_tokens
            
        elif self.type == "insertion":
            # Efficient insertion: count insertions per position
            insert_positions = np.random.choice(num_tokens + 1, n_tokens_to_edit, replace=True)
            insert_counts = np.zeros(num_tokens + 1, dtype=int)
            for pos in insert_positions:
                insert_counts[pos] += 1
                
            # Build the new tokens list by iterating once
            new_tokens = []
            for i in range(num_tokens + 1):
                if insert_counts[i] > 0:
                    # Generate all tokens to insert at this position at once
                    new_tokens.extend(
                        np.random.randint(0, self.vocab_size, insert_counts[i]).tolist()
                    )
                if i < num_tokens:
                    new_tokens.append(tokens[i])
            tokens = np.array(new_tokens)
            
        elif self.type == "deletion":
            selected_tokens_indices = np.random.choice(num_tokens, n_tokens_to_edit, replace=False)
            mask = np.ones(num_tokens, dtype=bool)
            mask[selected_tokens_indices] = False
            tokens = tokens[mask]
                
        edited_text = self.tokenizer.decode(tokens)
        return edited_text

            
        
        
class OpenAIAPI:
    """API class for OpenAI API."""
    def __init__(self, model, temperature, system_content):
        """
            Initialize OpenAI API with model, temperature, and system content.

            Parameters:
                model (str): Model name for OpenAI API.
                temperature (float): Temperature value for OpenAI API.
                system_content (str): System content for OpenAI API.
        """

        self.model = model
        self.temperature = temperature
        self.system_content = system_content
        self.client = openai.OpenAI()
        

        # List of supported models
        supported_models = ['gpt-3.5-turbo', 'gpt-4']

        # Check if the provided model is within the supported models
        if self.model not in supported_models:
            raise ValueError(f"Unsupported model '{self.model}'. Supported models are {supported_models}.")

    def get_result_from_gpt4(self, query):
        """get result from GPT-4 model."""
        response = self.client.chat.completions.create(
            model="gpt-4-0613",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response
    
    def get_result_from_gpt3_5(self, query):
        """get result from GPT-3.5 model."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": query},
            ]
        )
        return response

    def get_result(self, query):
        """get result from OpenAI API."""
        while True:
            try:
                if self.model == 'gpt-3.5-turbo':
                    result = self.get_result_from_gpt3_5(query)
                elif self.model == 'gpt-4':
                    result = self.get_result_from_gpt4(query)
                break
            except Exception as e:
                print(f"OpenAI API error: {str(e)}")
            time.sleep(10)
        return result.choices[0].message.content




class GPTParaphraser(TextEditor):
    """Paraphrase a text using the GPT model."""

    def __init__(self, openai_model: str, prompt: str) -> None:
        """
            Initialize the GPT paraphraser.

            Parameters:
                openai_model (str): The OpenAI model to use for paraphrasing.
                prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt

    def edit(self, text: str, reference=None):
        """Paraphrase the text using the GPT model."""

        openai_util = OpenAIAPI(model=self.openai_model, temperature=0.2, system_content="Your are a helpful assistant to rewrite the text.")
        paraphrased_text = openai_util.get_result(self.prompt + text)
        return paraphrased_text


class DipperParaphraser(TextEditor):
    """Paraphrase a text using the DIPPER model."""

    def __init__(self, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device='cuda',
                 lex_diversity: int = 60, order_diversity: int = 0, sent_interval: int = 1, **kwargs):
        """
            Paraphrase a text using the DIPPER model.

            Parameters:
                tokenizer (T5Tokenizer): The tokenizer for the DIPPER model.
                model (T5ForConditionalGeneration): The DIPPER model.
                device (str): The device to use for inference.
                lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
                sent_interval (int): The number of sentences to process at a time.
        """
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.lex_diversity = lex_diversity
        self.order_diversity = order_diversity
        self.sent_interval = sent_interval
        self.gen_kwargs = {}
        self.gen_kwargs.update(kwargs)

        # Validate diversity settings
        self._validate_diversity(self.lex_diversity, "Lexical")
        self._validate_diversity(self.order_diversity, "Order")
    
    def _validate_diversity(self, value: int, type_name: str):
        """Validate the diversity value."""
        if value not in [0, 20, 40, 60, 80, 100]:
            raise ValueError(f"{type_name} diversity must be a multiple of 20 from 0 to 100.")

    def edit(self, text: str, reference: str):
        """Edit the text using the DIPPER model."""

        # Calculate the lexical and order diversity codes
        lex_code = int(100 - self.lex_diversity)
        order_code = int(100 - self.order_diversity)
        
        # Preprocess the input text
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        
        # Preprocess the reference text
        prefix = " ".join(reference.replace("\n", " ").split())
        
        output_text = ""
        
        # Process the input text in sentence windows
        for sent_idx in range(0, len(sentences), self.sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + self.sent_interval])
            
            # Prepare the input for the model
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"
            
            # Tokenize the input
            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}
            
            # Generate the edited text
            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **self.gen_kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Update the prefix and output text
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the word deletion editor.

            Parameters:
                ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""

        # Handle empty string input
        if not text:  
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)

        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text


class ContextAwareSynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet based on the context."""

    def __init__(self, ratio: float, tokenizer: BertTokenizer, model: BertForMaskedLM, device='cuda') -> None:
        """
        Initialize the context-aware synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
            tokenizer (BertTokenizer): Tokenizer for BERT model.
            model (BertForMaskedLM): BERT model for masked language modeling.
            device (str): Device to run the model (e.g., 'cuda', 'cpu').
        """
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        nltk.download('wordnet')
    
    def _get_synonyms_from_wordnet(self, word: str):
        """ Return a list of synonyms for the given word using WordNet. """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet based on the context."""
        words = text.split()
        num_words = len(words)
        replaceable_indices = []

        for i, word in enumerate(words):
            if self._get_synonyms_from_wordnet(word):
                replaceable_indices.append(i)

        num_to_replace = int(min(self.ratio, len(replaceable_indices) / num_words) * num_words)
        indices_to_replace = random.sample(replaceable_indices, num_to_replace)

        real_replace = 0

        for i in indices_to_replace:
            # Create a sentence with a [MASK] token
            masked_sentence = words[:i] + ['[MASK]'] + words[i+1:]
            masked_text = " ".join(masked_sentence)
            
            # Use BERT to predict the token for [MASK]
            inputs = self.tokenizer(masked_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits[0, mask_position]
            predicted_indices = torch.argsort(predictions, descending=True)
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices[0:1])
            words[i] = predicted_tokens[0]
            real_replace += 1
        
        replaced_text = ' '.join(words)

        return replaced_text


class BackTranslationTextEditor(TextEditor):
    """Translate text from source language to intermediary language, then back to the source language."""

    def __init__(self,
                 translate_to_intermediary = Translator(from_lang="en", to_lang="zh").translate,
                 translate_to_source = Translator(from_lang="zh", to_lang="en").translate) -> None:
        """
        Initialize the back translation editor.

        Parameters:
            translate_to_intermediary (function): The function to translate text to the intermediary language.
            translate_to_source (function): The function to translate text to the source language.
        """
        super().__init__()
        self.translate_to_source = translate_to_source
        self.translate_to_intermediary = translate_to_intermediary

    def edit(self, text: str, reference=None):
        intermediary_text = self.translate_to_intermediary(text)
        edit_result = self.translate_to_source(intermediary_text)
        return edit_result