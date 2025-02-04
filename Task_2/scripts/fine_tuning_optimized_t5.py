import os
import logging
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    AdamW
)
import evaluate

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslationTrainer:
    def __init__(self, model_name="t5-small"):
        self.model_name = model_name
        self.output_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Los modelos se guardarán en: {self.output_dir}")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)

        # Activar gradient checkpointing para ahorrar memoria
        self.model.gradient_checkpointing_enable()

    def prepare_data(self, train_path, entities_path):
        """Optimizado con operaciones vectorizadas de pandas"""
        print("Cargando datasets...")
        train_df = pd.read_csv(train_path, sep=';')
        entities_df = pd.read_csv(entities_path, sep=';', names=['id', 'es', 'en'], header=0)

        entities_renamed = entities_df[['en', 'es']].rename(
            columns={'en': 'source', 'es': 'target'})
        combined_df = pd.concat([
            train_df[['source', 'target']],
            entities_renamed
        ], ignore_index=True)

        combined_df['source'] = 'translate English to Spanish: ' + combined_df['source']

        # Reducir longitud máxima para ahorrar memoria
        combined_df = combined_df[
            (combined_df['source'].str.len() <= 256) &  # Reducido de 512 a 256
            (combined_df['target'].str.len() <= 256)
        ].drop_duplicates().sample(frac=1, random_state=42)

        print(f"Dataset combinado final: {len(combined_df)} filas")

        train_data, val_data = train_test_split(
            combined_df,
            test_size=0.1,
            random_state=42
        )

        return (
            Dataset.from_pandas(train_data),
            Dataset.from_pandas(val_data)
        )

    def compute_metrics(self, eval_pred):
        metric = evaluate.load("sacrebleu")
        predictions, labels = eval_pred
        
        # Ensure predictions are within valid token range
        vocab_size = self.tokenizer.vocab_size
        predictions = np.clip(predictions, 0, vocab_size - 1)
        
        try:
            # Decode predictions with error handling
            decoded_preds = []
            for pred in predictions:
                try:
                    decoded = self.tokenizer.batch_decode(
                        [pred],
                        skip_special_tokens=True
                    )[0]
                    decoded_preds.append(decoded)
                except IndexError as e:
                    logging.warning(f"Error decoding prediction: {e}")
                    decoded_preds.append("")  # Add empty string for failed decodings
            
            # Handle labels similarly
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            labels = np.clip(labels, 0, vocab_size - 1)
            decoded_labels = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True
            )
            
            # Compute BLEU score
            bleu_score = metric.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )["score"]
            
            return {"bleu": bleu_score}
        
        except Exception as e:
            logging.error(f"Error in compute_metrics: {e}")
            return {"bleu": 0.0}  # Return default score in case of errors

    def preprocess_data(self, examples):
        source_texts = [text.strip() for text in examples['source']]
        target_texts = [text.strip() for text in examples['target']]

        model_inputs = self.tokenizer(
            source_texts,
            max_length=256,  # Reducido de 512 a 256
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            target_texts,
            max_length=256,  # Reducido de 512 a 256
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["labels"] = torch.where(
            model_inputs["labels"] == self.tokenizer.pad_token_id,
            -100,
            model_inputs["labels"]
        )
        return model_inputs

    def train(self, train_dataset, val_dataset):
        train_dataset = train_dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=val_dataset.column_names
        )

        final_model_dir = os.path.join(self.output_dir, "T5SmallFinetuningModel")
        checkpoint_dir = os.path.join(self.output_dir, "T5SmallFinetuningModel/checkpoints")

        # Argumentos optimizados para 4GB VRAM
        training_args = Seq2SeqTrainingArguments(
            output_dir=checkpoint_dir,
            evaluation_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            learning_rate=2e-5,
            per_device_train_batch_size=4,  # Reducido a 2 para menor uso de memoria
            per_device_eval_batch_size=4,   # Reducido a 2 para menor uso de memoria
            gradient_accumulation_steps=8,  # Aumentado para compensar batch size pequeño
            num_train_epochs=4,
            weight_decay=0.01,
            fp16=True,  # Activado fp16 para ahorrar memoria
            logging_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            save_total_limit=2,  # Reducido a 1 para ahorrar espacio
            predict_with_generate=True,
            generation_max_length=256,  # Reducido de 512 a 256
            generation_num_beams=2,
            optim="adamw_torch_fused",
            report_to="none",
            warmup_steps=1000,
            gradient_checkpointing=True,  # Activado para ahorrar memoria
            max_grad_norm=1.0  # Añadido para estabilidad
        )

        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8 if training_args.fp16 else None  # Optimización para fp16
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print("Iniciando entrenamiento...")
        trainer.train()

        print(f"Guardando modelo final en {final_model_dir}")
        trainer.save_model(final_model_dir)
        self.tokenizer.save_pretrained(final_model_dir)

        return trainer.evaluate()


def main():
    # Liberar memoria CUDA antes de empezar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer = TranslationTrainer()
    train_dataset, val_dataset = trainer.prepare_data(
        'data/processed/train/train_with_labels_cleaned.csv',
        'data/processed/train/entities_cleaned.csv'
    )
    results = trainer.train(train_dataset, val_dataset)

    print("\nResultados finales:")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()