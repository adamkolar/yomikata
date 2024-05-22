"""
dbert.py
Provides the dBert class that implements Reader using BERT contextual embeddings to disambiguate heteronyms.
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from speach.ttlig import RubyFrag, RubyToken
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    AutoConfig
)

from yomikata.custom_bert import CustomBertForTokenClassification, CustomDataCollatorForTokenClassification
from yomikata import utils
from yomikata.config import config, logger
from yomikata.reader import Reader
from yomikata.utils import LabelEncoder
import evaluate
import jaconv
from collections import defaultdict
import math
import random

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


class dBert(Reader):
    def __init__(
        self,
        artifacts_dir: Path = config.DBERT_DIR,
        reinitialize: bool = False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        # Set the device
        self.device = device
        logger.info(f"Running on {self.device}")
        if self.device.type == "cuda":
            logger.info(torch.cuda.get_device_name(0))

        # Hardcoded parameters
        self.max_length = 128

        # Load the model
        self.artifacts_dir = artifacts_dir

        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            reinitialize = True

        if reinitialize:
            # load tokenizer from upstream huggingface repository
            default_model = "cl-tohoku/bert-base-japanese-char-v3"
            self.tokenizer = AutoTokenizer.from_pretrained(default_model)
            logger.info(f"Using {default_model} tokenizer")

            # load the heteronyms list
            self.heteronyms = config.HETERONYMS

            label_list = {}
            for i, heteronym in enumerate(self.heteronyms.keys()):
                label_list[heteronym] = []
                for j, reading in enumerate(self.heteronyms[heteronym]):
                    label_list[heteronym].append(reading)

            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(label_list)

            logger.info("Made label encoder with default heteronyms")

            cfg = AutoConfig.from_pretrained(default_model)
            dropout_rate = 0.1 # You can adjust this value
            cfg.hidden_dropout_prob = dropout_rate
            cfg.attention_probs_dropout_prob = dropout_rate
            cfg.num_labels = len(self.label_encoder.classes)

            # Load model from upstream huggingface repository
            self.model = CustomBertForTokenClassification.from_pretrained(default_model, config=cfg)

            logger.info(f"Using model {default_model}")

            self.save(artifacts_dir)
        else:
            self.load(artifacts_dir)

    def load(self, directory):
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.model = CustomBertForTokenClassification.from_pretrained(directory).to(self.device)
        self.label_encoder = LabelEncoder.load(Path(directory, "label_encoder.json"))
        self.heteronyms = utils.load_dict(Path(directory, "heteronyms.json"))
        logger.info(f"Loaded model from directory {directory}")

    def save(self, directory):
        self.tokenizer.save_pretrained(directory)
        self.model.save_pretrained(directory)
        self.label_encoder.save(Path(directory, "label_encoder.json"))
        utils.save_dict(self.heteronyms, Path(directory, "heteronyms.json"))
        logger.info(f"Saved model to directory {directory}")

    def batch_preprocess_function(self, entries, pad=False):
        tokenized_inputs = self.tokenizer(
            entries["sentence"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length" if pad else "do_not_pad",
        )

        labels = []
        valid_class_masks = []
        class_map = self.label_encoder.class_to_index
        surfaces = list(set(x.split(":")[0] for x in class_map.keys()))
        group_boundaries = self.label_encoder.group_boundaries
        for input_ids, sentence, furigana in zip(tokenized_inputs["input_ids"], entries["sentence"], entries["furigana"]):
            matches = utils.find_all_substrings(sentence, surfaces)
            furis = utils.get_furis(furigana)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            location = 0
            label_ids = []
            class_mask = []
            for token in tokens:
                label = -100
                current_mask = 0
                if token not in self.tokenizer.all_special_tokens:
                    token = token[2:] if token[:2] == "##" else token
                    if location in matches:
                        surface = matches[location]
                        reading = utils.get_reading_from_furi(location, len(surface), furis)
                        if reading is not None:
                            reading = jaconv.kata2hira(reading)
                            class_name = f"{surface}:{reading}"
                            if class_name in class_map:
                                label = class_map[class_name]
                                current_mask = (group_boundaries[surface][0] << 32) | group_boundaries[surface][1]
                    location += len(token)
                class_mask.append(current_mask)
                label_ids.append(label)
            labels.append(label_ids)
            valid_class_masks.append(class_mask)
        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
            "valid_mask": valid_class_masks
        }

    def train(self, dataset, training_args={}) -> dict:

        dataset = dataset.map(
            self.batch_preprocess_function, batched=True, fn_kwargs={"pad": False}
        )
        dataset = dataset.filter(
            lambda entry: any(label != -100 for label in entry["labels"])
        )

        class_counts = defaultdict(int)

        def update_class_counts(example):
            for label in example["labels"]:
                if label != -100:
                    class_counts[label] += 1

        dataset["train"].map(
            update_class_counts,
            with_indices=False, load_from_cache_file=False
        )

        n_classes = len(self.label_encoder.class_to_index)
        alpha_values = [0] * n_classes
        total_samples = sum(class_counts.values())
        for class_id, count in class_counts.items():
            alpha_values[class_id] = math.sqrt(total_samples) / math.sqrt(count)
        sum_alpha = sum(alpha_values) + 1e-10
        normalized_alpha_values = [alpha / sum_alpha for alpha in alpha_values]
        alpha_tensor = torch.tensor(normalized_alpha_values, dtype=torch.float)

        # put the model in training mode
        self.model.train()

        self.model.set_alpha(alpha_tensor)

        batch_size = 64
        default_training_args = {
            "output_dir": self.artifacts_dir,
            "num_train_epochs": 10,
            "evaluation_strategy": "epoch",
            "logging_strategy": "epoch",
            "save_strategy": "epoch",
            "learning_rate": 5e-5,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "load_best_model_at_end": True,
            "metric_for_best_model": "accuracy",
            "weight_decay": 0.01,
            "save_total_limit": 3,
            "fp16": True,
            "report_to": "mlflow",
            "disable_tqdm": False,
            "eval_accumulation_steps": 100,
            "include_inputs_for_metrics": False,
        }

        default_training_args.update(training_args)

        # if default_training_args["evaluation_strategy"] == "steps" and "eval_steps" not in training_args:
        #     custom_batch_size = training_args["per_device_train_batch_size"] if "per_device_train_batch_size" in training_args else batch_size
        #     stops_per_epoch = 2
        #     steps = round(len(dataset["train"])/stops_per_epoch/custom_batch_size)
        #     default_training_args["eval_steps"] = steps
        #     default_training_args["logging_steps"] = steps
        #     default_training_args["save_steps"] = steps

        training_args = default_training_args

        # Not padding in batch_preprocess_function so need data_collator for trainer
        data_collator = CustomDataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)

        accuracy_metric = evaluate.load("accuracy")
        recall_metric = evaluate.load("recall")

        def compute_metrics(p):
            predictions, labels = p  # predictions are already the argmax of logits
            true_predictions = [pred for prediction, label in zip(predictions, labels) for pred, lab in zip(prediction, label) if lab != -100]
            true_labels = [lab for prediction, label in zip(predictions, labels) for pred, lab in zip(prediction, label) if lab != -100]
            return {"accuracy": accuracy_metric.compute(references=true_labels, predictions=true_predictions)["accuracy"], "recall": recall_metric.compute(references=true_labels, predictions=true_predictions, average="macro", zero_division=0)["recall"]}

        if "val" in list(dataset):
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args),
                train_dataset=dataset["train"],
                eval_dataset=dataset["val"],
                tokenizer=self.tokenizer,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=3),
                ],
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=lambda logits, _: torch.argmax(logits, dim=-1)
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=TrainingArguments(**training_args),
                train_dataset=dataset["train"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                preprocess_logits_for_metrics=lambda logits, _: torch.argmax(logits, dim=-1)
            )

        result = trainer.train()

        # Output some training information
        print(f"Time: {result.metrics['train_runtime']:.2f}")
        print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
        gpu_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
        utils.print_gpu_utilization(gpu_index)
        trainer.save_model()

        # Get metrics for each train/val/split
        self.model.eval()
        full_performance = {}
        for key in dataset.keys():
            max_evals = min(100000, len(dataset[key]))
            # max_evals = len(dataset[key])
            logger.info(f"getting predictions for {key}")
            subset = dataset[key].shuffle().select(range(max_evals))
            prediction_output = trainer.predict(subset)
            logger.info(f"processing predictions for {key}")
            metrics = prediction_output[2]
            labels = prediction_output[1]

            logger.info("processing performance")
            performance = {
                heteronym: {
                    "n": 0,
                    "readings": {
                        reading: {
                            "n": 0,
                            "found": {readingprime: 0 for readingprime in list(self.heteronyms[heteronym].keys())}
                        }
                        for reading in list(self.heteronyms[heteronym].keys())
                    },
                }
                for heteronym in self.heteronyms.keys()
            }

            flattened_logits = [
                logit
                for sequence_logits, sequence_labels in zip(prediction_output[0], labels)
                for (logit, l) in zip(sequence_logits, sequence_labels) if l != -100
            ] # this is already argmaxed in preprocess_logits_for_metrics, so the resulting list is 1d. valid_mask processing in CustomBertForTokenClassification.forward takes care of zeoring out irrelevant logits

            true_labels = [
                str(self.label_encoder.index_to_class[l])
                for label in labels
                for l in label if l != -100
            ]

            for i, true_label in enumerate(true_labels):
                (true_surface, true_reading) = true_label.split(":")
                performance[true_surface]["n"] += 1
                performance[true_surface]["readings"][true_reading]["n"] += 1
                predicted_label = self.label_encoder.index_to_class[flattened_logits[i]]
                predicted_reading = predicted_label.split(":")[1]
                performance[true_surface]["readings"][true_reading]["found"][predicted_reading] += 1

            for surface in performance:
                for true_reading in performance[surface]["readings"]:
                    true_count = performance[surface]["readings"][true_reading]["n"]
                    predicted_count = performance[surface]["readings"][true_reading]["found"][true_reading]
                    performance[surface]["readings"][true_reading]["accuracy"] = predicted_count / true_count if true_count > 0 else "NaN"
                correct_count = sum(performance[surface]["readings"][true_reading]["found"][true_reading] for true_reading in performance[surface]["readings"])
                all_count = performance[surface]["n"]
                performance[surface]["accuracy"] = correct_count / all_count if all_count > 0 else "NaN"

            performance = {
                "metrics": metrics,
                "heteronym_performance": performance,
            }

            full_performance[key] = performance

        return full_performance

    def furigana(self, text: str) -> str:
        text = utils.standardize_text(text)
        text = utils.remove_furigana(text)
        text = text.replace("{", "").replace("}", "")

        self.model.eval()

        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_encoded["input_ids"].to(self.device)
        input_mask = text_encoded["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=input_mask).logits

        surfaces = list(set([x.split(":")[0] for x in self.label_encoder.classes]))
        matches = utils.find_all_substrings(text, surfaces)

        location = 0

        for i, prediction in enumerate(logits[0]):
            input_id = input_ids[0][i]
            token_text = self.tokenizer.decode([input_id])
            if token_text in self.tokenizer.all_special_tokens:
                continue
            if token_text[:2] == "##":
                token_text = token_text[2:]
            if location in matches:
                surface = matches[location]
                valid_indexes = [idx for idx, s in enumerate(self.label_encoder.classes) if s.split(":")[0] == surface]
                valid_logits = logits[0][i][valid_indexes]
                rel_prediction = torch.argmax(valid_logits).item()
                abs_prediction = valid_indexes[rel_prediction]
                matches[location] = self.label_encoder.index_to_class[abs_prediction]
            location += len(token_text)

        output_ruby = []
        last_loc = 0
        for loc in matches:
            if loc != last_loc:
                output_ruby.append(text[last_loc:loc])
            match = matches[loc]
            if ":" in match:
                match_split = match.split(":")
                last_loc = loc + len(match_split[0])
                output_ruby.append(RubyFrag(text=match_split[0], furi=match_split[1]))
            else:
                last_loc = loc + len(match)
                output_ruby.append(f"{{{match}}}")

        if last_loc < len(text):
            output_ruby.append(text[last_loc:])


        return RubyToken(groups=output_ruby).to_code()
