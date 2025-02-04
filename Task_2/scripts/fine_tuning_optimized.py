import os
import logging
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
	MarianTokenizer,
	MarianMTModel,
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
	def __init__(self, model_name="Helsinki-NLP/opus-mt-en-es"):
		self.model_name = model_name
		self.output_dir = os.path.join(os.getcwd(), "models")
		os.makedirs(self.output_dir, exist_ok=True)

		print(f"Los modelos se guardarán en: {self.output_dir}")

		self.tokenizer = MarianTokenizer.from_pretrained(model_name)
		self.model = MarianMTModel.from_pretrained(
			model_name,
			torch_dtype=torch.float32
		).to(device)

		if torch.cuda.is_available():
			self.model.gradient_checkpointing_enable()

	def prepare_data(self, train_path, entities_path):
		"""Optimizado con operaciones vectorizadas de pandas"""
		# Cargar datasets
		print("Cargando datasets...")
		train_df = pd.read_csv(train_path, sep=';')
		entities_df = pd.read_csv(entities_path, sep=';', names=[
								  'id', 'es', 'en'], header=0)

		# Combinación más eficiente
		entities_renamed = entities_df[['en', 'es']].rename(
			columns={'en': 'source', 'es': 'target'})
		combined_df = pd.concat([
			train_df[['source', 'target']],
			entities_renamed
		], ignore_index=True)

		# Filtrado y limpieza
		combined_df = combined_df[
			(combined_df['source'].str.len() <= 128) &
			(combined_df['target'].str.len() <= 128)
		].drop_duplicates().sample(frac=1, random_state=42)  # Mezclar datos

		print(f"Dataset combinado final: {len(combined_df)} filas")

		# Split estratificado
		train_data, val_data = train_test_split(
			combined_df,
			test_size=0.1,  # Reducido para más datos de entrenamiento
			random_state=42
		)

		return (
			Dataset.from_pandas(train_data),
			Dataset.from_pandas(val_data)
		)

	def compute_metrics(self, eval_pred):
		metric = evaluate.load("sacrebleu")
		predictions, labels = eval_pred

		decoded_preds = self.tokenizer.batch_decode(
			predictions,
			skip_special_tokens=True
		)

		labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
		decoded_labels = self.tokenizer.batch_decode(
			labels,
			skip_special_tokens=True
		)

		return {"bleu": metric.compute(
				predictions=decoded_preds,
				references=[[label] for label in decoded_labels]
				)["score"]}

	def preprocess_data(self, examples):
		"""Preprocesado con padding dinámico"""
		source_texts = [text.strip() for text in examples['source']]
		target_texts = [text.strip() for text in examples['target']]

		model_inputs = self.tokenizer(
			source_texts,
			max_length=128,
			padding='max_length',
			truncation=True,
			return_tensors="pt"
		)

		with self.tokenizer.as_target_tokenizer():
			labels = self.tokenizer(
				target_texts,
				max_length=128,
				padding='max_length',
				truncation=True,
				return_tensors="pt"
			)

		model_inputs["labels"] = labels["input_ids"]
		return model_inputs

	def train(self, train_dataset, val_dataset):
		"""Configura y ejecuta el entrenamiento"""
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

		final_model_dir = os.path.join(self.output_dir, "FinetuningModel")
		checkpoint_dir = os.path.join(
			self.output_dir, "FinetuningModel/checkpoints")

		training_args = Seq2SeqTrainingArguments(
			output_dir=checkpoint_dir,
			evaluation_strategy="steps",
			eval_steps=3000,  # Evaluación más frecuente
			save_strategy="steps",
			save_steps=3000,
			learning_rate=3e-5,  # Tasa de aprendizaje ajustada
			per_device_train_batch_size=4,  # Batch size aumentado
			per_device_eval_batch_size=4,
			gradient_accumulation_steps=8,  # Pasos de acumulación reducidos
			num_train_epochs=4,  # Épocas aumentadas
			weight_decay=0.01,
			fp16=False,
			logging_steps=500,
			load_best_model_at_end=True,
			metric_for_best_model="bleu",
			greater_is_better=True,
			save_total_limit=1,  # Menor uso de almacenamiento
			predict_with_generate=True,
			generation_max_length=128,
			generation_num_beams=2,  # Beam search reducido
			optim="adamw_torch_fused",  # Optimizador más eficiente en memoria
			report_to="none"  # Deshabilitar reportes externos
		)

		data_collator = DataCollatorForSeq2Seq(
			self.tokenizer,
			model=self.model,
			pad_to_multiple_of=None
		)

		trainer = Seq2SeqTrainer(
			model=self.model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=val_dataset,
			data_collator=data_collator,
			tokenizer=self.tokenizer,
			compute_metrics=self.compute_metrics,
			callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
		)

		print("Iniciando entrenamiento...")
		trainer.train()

		print(f"Guardando modelo final en {final_model_dir}")
		trainer.save_model(final_model_dir)
		self.tokenizer.save_pretrained(final_model_dir)

		return trainer.evaluate()


def main():
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
