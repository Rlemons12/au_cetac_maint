import os
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# Canonical directories come from config (single source of truth)
from modules.configuration.config import ORC_INTENT_MODEL_DIR, ORC_PARTS_MODEL_DIR  # change NER default if needed
from modules.configuration.log_config import logger


class IntentEntityPlugin:
    def __init__(self, intent_model_dir=None, ner_model_dir=None, intent_labels=None, ner_labels=None):
        """
        intent_model_dir: Path to intent classifier model directory
        ner_model_dir: Path to NER model directory
        """

        # ---- DIAGNOSTIC: caller + raw args + config defaults ----
        try:
            caller = inspect.stack()[1]
            logger.debug(f"[IntentEntityPlugin] LOADED FROM: {inspect.getfile(IntentEntityPlugin)}")
            logger.debug(f"[IntentEntityPlugin] CALLED BY: {caller.filename}:{caller.lineno}")
        except Exception:
            pass
        logger.debug(f"[IntentEntityPlugin] ARG intent_model_dir={intent_model_dir!r}, ner_model_dir={ner_model_dir!r}")
        logger.debug(f"[IntentEntityPlugin] CFG ORC_INTENT_MODEL_DIR={ORC_INTENT_MODEL_DIR}")
        logger.debug(f"[IntentEntityPlugin] CFG ORC_PARTS_MODEL_DIR={ORC_PARTS_MODEL_DIR}")

        # ---- Resolve to config defaults & coerce legacy literals ----
        LEGACY_INTENT = {"models/intent", "intent", "./models/intent", ".\\models\\intent"}
        LEGACY_NER    = {"models/ner", "ner", "./models/ner", ".\\models\\ner"}

        if not intent_model_dir or intent_model_dir in LEGACY_INTENT:
            intent_model_dir = ORC_INTENT_MODEL_DIR
        if not ner_model_dir or ner_model_dir in LEGACY_NER:
            ner_model_dir = ORC_PARTS_MODEL_DIR  # TODO: replace with ORC_NER_MODEL_DIR if you have one

        self.intent_model_dir = intent_model_dir
        self.ner_model_dir    = ner_model_dir

        logger.debug(f"[IntentEntityPlugin] RESOLVED intent_model_dir={self.intent_model_dir}")
        logger.debug(f"[IntentEntityPlugin] RESOLVED ner_model_dir={self.ner_model_dir}")

        # ---- Back-compat redirect: .../models/intent -> .../models/intent_classifier ----
        if os.path.basename(self.intent_model_dir.rstrip("/\\")) == "intent":
            candidate = os.path.join(os.path.dirname(self.intent_model_dir), "intent_classifier")
            if os.path.isdir(candidate):
                logger.info(f"[IntentEntityPlugin] Redirecting 'intent' -> '{candidate}'")
                self.intent_model_dir = candidate

        # ---- Ensure folders exist (prevents scandir crashes) ----
        if not os.path.isdir(self.intent_model_dir):
            os.makedirs(self.intent_model_dir, exist_ok=True)
            logger.warning(f"[IntentEntityPlugin] Created missing intent model dir: {self.intent_model_dir}")

        if self.ner_model_dir and not os.path.isdir(self.ner_model_dir):
            os.makedirs(self.ner_model_dir, exist_ok=True)
            logger.warning(f"[IntentEntityPlugin] Created missing NER model dir: {self.ner_model_dir}")

        # ---- Auto-detect checkpoint subfolders (if base lacks config.json) ----
        if os.path.isdir(self.intent_model_dir) and not os.path.exists(os.path.join(self.intent_model_dir, "config.json")):
            try:
                for f in os.scandir(self.intent_model_dir):
                    if f.is_dir() and os.path.exists(os.path.join(f.path, "config.json")):
                        logger.info(f"[IntentEntityPlugin] Auto-detected intent checkpoint: {f.path}")
                        self.intent_model_dir = f.path
                        break
            except Exception as e:
                logger.warning(f"[IntentEntityPlugin] Unable to scan intent dir '{self.intent_model_dir}': {e}")

        if self.ner_model_dir and os.path.isdir(self.ner_model_dir) and not os.path.exists(os.path.join(self.ner_model_dir, "config.json")):
            try:
                for f in os.scandir(self.ner_model_dir):
                    if f.is_dir() and os.path.exists(os.path.join(f.path, "config.json")):
                        logger.info(f"[IntentEntityPlugin] Auto-detected NER checkpoint: {f.path}")
                        self.ner_model_dir = f.path
                        break
            except Exception as e:
                logger.warning(f"[IntentEntityPlugin] Unable to scan NER dir '{self.ner_model_dir}': {e}")

        # ---- labels ----
        self.intent_labels   = intent_labels or ["parts", "images", "documents", "prints", "tools", "troubleshooting"]
        self.intent_id2label = {i: label for i, label in enumerate(self.intent_labels)}
        self.intent_label2id = {label: i for i, label in enumerate(self.intent_labels)}

        self.ner_labels   = ner_labels or ["O", "B-PARTDESC", "B-PARTNUM"]
        self.ner_id2label = {i: label for i, label in enumerate(self.ner_labels)}
        self.ner_label2id = {label: i for i, label in enumerate(self.ner_labels)}

        # ---- lazy pipelines ----
        self.intent_classifier = None
        self.ner = None

        # ---- Load intent pipeline if files exist ----
        if os.path.isdir(self.intent_model_dir):
            try:
                cfg = os.path.join(self.intent_model_dir, "config.json")
                weights = [f for f in os.listdir(self.intent_model_dir) if f.endswith((".bin", ".safetensors"))]
                if os.path.exists(cfg) and weights:
                    # Preload locally, then feed into pipeline (avoids passing local_files_only to pipeline)
                    intent_tokenizer = AutoTokenizer.from_pretrained(self.intent_model_dir, local_files_only=True,
                                                                     trust_remote_code=False)
                    intent_model = AutoModelForSequenceClassification.from_pretrained(
                        self.intent_model_dir,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                    self.intent_classifier = pipeline(
                        "text-classification",
                        model=intent_model,
                        tokenizer=intent_tokenizer,  # <-- fix
                        trust_remote_code=False
                    )

                else:
                    logger.warning(f"[IntentEntityPlugin] Intent model files not found in {self.intent_model_dir}")
            except Exception as e:
                logger.warning(f"[IntentEntityPlugin] Could not load intent classifier: {e}")

        # ---- Load NER pipeline if files exist ----
        if self.ner_model_dir and os.path.isdir(self.ner_model_dir):
            try:
                cfg = os.path.join(self.ner_model_dir, "config.json")
                weights = [f for f in os.listdir(self.ner_model_dir) if f.endswith((".bin", ".safetensors"))]
                if os.path.exists(cfg) and weights:
                    ner_tokenizer = AutoTokenizer.from_pretrained(self.ner_model_dir, local_files_only=True,
                                                                  trust_remote_code=False)
                    ner_model = AutoModelForTokenClassification.from_pretrained(
                        self.ner_model_dir,
                        local_files_only=True,
                        trust_remote_code=False
                    )
                    self.ner = pipeline(
                        "ner",
                        model=ner_model,
                        tokenizer=ner_tokenizer,
                        aggregation_strategy="simple",
                        # no local_files_only here
                        trust_remote_code=False
                    )
                else:
                    logger.warning(f"[IntentEntityPlugin] NER model files not found in {self.ner_model_dir}")
            except Exception as e:
                logger.warning(f"[IntentEntityPlugin] Could not load NER model: {e}")
    def classify_intent(self, text: str):
        """
        Runs the intent classifier on the input text and returns
        (predicted_intent, confidence_score)
        """
        if not self.intent_classifier:
            logger.warning("[IntentEntityPlugin] Intent classifier not initialized.")
            return None, 0.0

        try:
            results = self.intent_classifier(text)
            if results and len(results) > 0:
                top = results[0]
                label = top["label"]
                score = float(top["score"])
                return label, score
        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] Intent classification failed: {e}")

        return None, 0.0

    def extract_entities(self, text: str):
        """
        Runs the NER model on the input text and returns
        a list of entities with labels.
        """
        if not self.ner:
            logger.warning("[IntentEntityPlugin] NER model not initialized.")
            return []

        try:
            results = self.ner(text)
            entities = []
            for ent in results:
                entities.append({
                    "entity": ent["word"],
                    "label": ent["entity_group"],
                    "score": float(ent["score"])
                })
            return entities
        except Exception as e:
            logger.warning(f"[IntentEntityPlugin] NER extraction failed: {e}")
            return []

