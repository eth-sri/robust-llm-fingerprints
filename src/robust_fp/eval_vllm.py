# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from robust_fp.config import FingerprintEvalConfiguration
from robust_fp.eval.text_editors import TextEditor
from typing import Optional, List, Dict
import json
import os
from robust_fp.quality.llm_judge import get_gpt4_grades
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from copy import deepcopy
from datasets import load_dataset, load_from_disk


class EvaluationVLLM:
    def __init__(
        self,
        fingerprint_eval_config: FingerprintEvalConfiguration,
        overwrite: Optional[bool] = False,
        output_dir: Optional[str] = "output",
    ):

        self.fingerprint_eval_config = fingerprint_eval_config
        self.overwrite = overwrite
        self.output_dir = output_dir
        self.run_number = 0

    def check_if_results_exist(self, name):
        if self.overwrite:
            return False
        res_path = f"{self.output_dir}/{self.run_number}/results_{name}.jsonl"
        return os.path.exists(res_path)

    def save_results(
        self,
        results: Dict[str, List],
        name: str,
    ):
        res_path = f"{self.output_dir}/{self.run_number}/results_{name}.jsonl"
        os.makedirs(os.path.dirname(res_path), exist_ok=True)

        # Save results in JSONL format
        with open(res_path, "w") as file:
            for values in zip(*results.values()):
                line_dict = {key: value for key, value in zip(results.keys(), values)}
                file.write(json.dumps(line_dict) + "\n")


    def get_fp_eval_dataset(self, fingerprint_eval_config):
        dataset_config = deepcopy(fingerprint_eval_config.dataset_config)

        is_chat = "data_fields" in dataset_config
        is_local = dataset_config.pop("is_local", False)

        if is_chat:
            question_field, _ = dataset_config.pop("data_fields")
            special_token = dataset_config.pop("special_token", "")
        else:
            data_field = dataset_config.pop("data_field")
            special_token = dataset_config.pop("special_token", "")

        if is_local:
            dataset = load_from_disk(**dataset_config)
        else:
            dataset = load_dataset(
                **dataset_config, trust_remote_code=True, streaming=True
            )

        prompts = []
        if is_chat:
            for i, row in enumerate(dataset):
                conversation = [
                    {"role": "user", "content": special_token + row[question_field]},
                ]
                prompts.append(conversation)

                if i == fingerprint_eval_config.n_samples - 1:
                    break
        else:
            for i, row in enumerate(dataset):
                prompts.append(special_token + row[data_field])

                if i == fingerprint_eval_config.n_samples - 1:
                    break

        return prompts, is_chat

    def eval_model(
        self,
        model: LLM,
        sampling_parameters: SamplingParams,
        input_editor: TextEditor,
        output_editor: TextEditor,
        lora_request: LoRARequest,
    ):
        check = self.check_if_results_exist(self.fingerprint_eval_config.name)
        if check:
            print(f"Results for {self.fingerprint_eval_config.name} already exist, skipping")
            return

        res = self._eval_fingerprint(
            model, sampling_parameters, input_editor, output_editor, lora_request, self.fingerprint_eval_config
        )

        self.save_results(
            results=res,
            name=self.fingerprint_eval_config.name,
        )

    def _eval_fingerprint(
        self, 
        model: LLM, 
        sampling_parameters: SamplingParams,
        input_editor: TextEditor,
        output_editor: TextEditor,
        lora_request: LoRARequest,
        fingerprint_eval_config: FingerprintEvalConfiguration,
    ):
        prompts_ds, is_chat = self.get_fp_eval_dataset(fingerprint_eval_config)

        if input_editor is not None:
            prompts_ds = input_editor.edit_input(prompts_ds)

        additional_params = input_editor.get_additional_params() if input_editor else {}

        if is_chat:
            outputs = model.chat(prompts_ds, sampling_parameters, lora_request=lora_request, **additional_params)
        else:
            outputs = model.generate(prompts_ds, sampling_parameters, lora_request=lora_request, **additional_params)

        output_texts = [output.outputs[0].text for output in outputs]
        if output_editor is not None:
            output_texts = output_editor.edit_output(output_texts)
        prompts = []
        completions = []
        for i, output in enumerate(output_texts):
            if is_chat:
                if not additional_params.get("continue_final_message", False):
                    prompts.append(prompts_ds[i][-1]["content"])
                else:
                    prompts.append(
                        prompts_ds[i][-2]["content"]
                        + "\n\n"
                        + prompts_ds[i][-1]["content"]
                    )
            else:
                prompts.append(prompts_ds[i])
            completions.append(output)
        res = {
            "prompts": prompts,
            "completions": completions,
        }

        return res

    def get_gpt4_score(
        self,
        prompts: List[str],
        completions: List[str],
        is_completion_task: bool = False,
    ) -> List[float]:
        gpt4_scores = get_gpt4_grades(prompts, completions, is_completion_task)
        scores = []
        explanations = []

        for i, score_dict in enumerate(gpt4_scores):
            explanations.append(score_dict)

            comb_score = 0
            ctr = 0
            for key, val in score_dict.items():
                if key != "ethics":
                    if val["grade"] == -1:
                        continue
                    comb_score += val["grade"]
                    ctr += 1

            comb_score /= max(ctr, 1.0)

            scores.append(comb_score)

        return scores, explanations

