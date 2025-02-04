import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    MarianTokenizer,
    MarianMTModel,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

# Configuración general
logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "models/Translation/"
FINAL_MODEL_DIR = "models/modelo_traduccion_finetuned/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    try:
        # Cargar datasets (ajustar rutas según necesidad)
        train_df = pd.read_csv('data/processed/train/train_cleaned_source.csv', sep=';')
        entities_df = pd.read_csv('data/processed/train/entities_cleaned.csv', 
                                sep=';', header=None, names=['id', 'es', 'en'])
        
        # Aumentación de datos
        additional_data = []
        for _, row in entities_df.iterrows():
            # Asegurarse de que 'en' y 'es' son cadenas
            en_text = str(row['en']) if isinstance(row['en'], (str, bytes)) else ""
            es_text = str(row['es']) if isinstance(row['es'], (str, bytes)) else ""
            additional_data.extend([
                {"source": en_text, "target": es_text},
                {"source": en_text.lower(), "target": es_text.lower()},
                {"source": en_text.upper(), "target": es_text.upper()}
            ])
            
        # Combinar y balancear
        enriched_df = pd.concat([
            train_df[['source', 'target']],
            pd.DataFrame(additional_data)
        ], ignore_index=True).drop_duplicates().sample(frac=1, random_state=42)
        
        # Validación
        assert not enriched_df.isnull().values.any(), "Datos con valores nulos"
        
        # División estratificada
        train_data, val_data = train_test_split(
            enriched_df,
            test_size=0.15,
            random_state=42,
            stratify=pd.qcut(enriched_df['source'].apply(len), q=5)
        )
        
        return Dataset.from_pandas(train_data), Dataset.from_pandas(val_data)
    
    except Exception as e:
        logging.error(f"Error en carga de datos: {e}")
        raise
    
train_dataset, val_dataset = load_and_preprocess_data()
print(f"📊 Datos de entrenamiento: {len(train_dataset)} ejemplos")
print(f"📊 Datos de validación: {len(val_dataset)} ejemplos")

def setup_model():
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    
    # Configuración especial para entrenamiento
    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(["</s>"])[0]  # <s> es el token de inicio
    model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(["</s>"])[0]  # <s> es el token de inicio
    model.config.max_length = 256  # Aumentar longitud máxima
    return tokenizer, model

tokenizer, model = setup_model()
print("✅ Modelo y tokenizador cargados correctamente")

def tokenize_batch(batch):
    inputs = tokenizer(
        batch["source"],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )
    
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(
            batch["target"],
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
    
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": targets["input_ids"].squeeze()
    }

tokenized_train = train_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1024,
    remove_columns=train_dataset.column_names
)

tokenized_val = val_dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=1024,
    remove_columns=val_dataset.column_names
)

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(f"🚀 Paso {state.global_step} - Pérdida: {logs.get('loss', 0):.4f}")

def create_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        bleu = evaluate.load("bleu")
        sacrebleu = evaluate.load("sacrebleu")
        
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        return {
            "bleu": bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["bleu"],
            "sacrebleu": sacrebleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])["score"]
        }
    return compute_metrics

# Verificar checkpoints previos
checkpoint = get_last_checkpoint(MODEL_DIR) if os.path.exists(MODEL_DIR) else None

# Configurar arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    eval_steps=500,
    save_strategy="epoch",
    save_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=500,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    greater_is_better=True,
    predict_with_generate=True,
    report_to="tensorboard",
    resume_from_checkpoint=checkpoint
)

# Inicializar Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=create_compute_metrics(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ProgressCallback()]
)

try:
    print("🏋️ Iniciando entrenamiento...")
    print(f"🔍 Checkpoint inicial: {checkpoint or 'Ninguno'}")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    
    print("💾 Guardando modelo final...")
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print("🧪 Evaluación final:")
    results = trainer.evaluate()
    print(f"✅ BLEU: {results['eval_bleu']:.2f}")
    print(f"✅ SacreBLEU: {results['eval_sacrebleu']:.2f}")
except Exception as e:
    print(f"❌ Error durante el entrenamiento: {e}")
    raise
