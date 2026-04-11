"""
Machine learning-based legal text classification into domain categories.

Uses scikit-learn with TF-IDF vectorization and Random Forest classification
to categorize legal text into Constitutional, Criminal, Contract, or Other domains.
"""

import pickle
import numpy as np
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)


class LegalTextClassifier:
    """Machine learning classifier for legal text domain categorization."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.is_trained = False
        self.categories = {
            0: "Constitutional Law",
            1: "Criminal Law", 
            2: "Contract Law",
            3: "Other"
        }
    
    def _create_training_data(self):
        import asyncio
        from pathlib import Path
        import sys
        
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        try:
            from app.core.database import db_manager
            
            async def get_document_training_data():
                try:
                    await db_manager.initialize()
                    async with db_manager.get_connection() as conn:
                        result = await conn.fetch("""
                            SELECT content, source, title 
                            FROM documents 
                            WHERE content IS NOT NULL 
                            AND LENGTH(content) > 100
                            LIMIT 50
                        """)
                        
                        if result:
                            texts = []
                            labels = []
                            
                            for row in result:
                                content = row['content']
                                source = row['source'] or ''
                                title = row['title'] or ''
                                
                                label = self._classify_document_content(content, source, title)
                                texts.append(content[:500])
                                labels.append(label)
                            
                            return {"texts": texts, "labels": labels}
                except Exception as e:
                    logger.warning(f"Could not get training data from database: {e}")
                    return None
            
            # Handle async context - can't run async code if already in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    training_data = None
                else:
                    training_data = asyncio.run(get_document_training_data())
            except Exception as e:
                logger.warning(f"Could not get training data from database: {e}")
                training_data = None
            
            if not training_data or len(training_data["texts"]) < 10:
                logger.info("Using fallback training data - no sufficient documents in database")
                training_data = self._get_fallback_training_data()
            
            return training_data
            
        except Exception as e:
            logger.warning(f"Error creating training data: {e}, using fallback")
            return self._get_fallback_training_data()
    
    def _classify_document_content(self, content: str, source: str, title: str) -> int:
        content_lower = content.lower()
        source_lower = source.lower()
        title_lower = title.lower()
        
        constitutional_keywords = [
            'constitution', 'amendment', 'bill of rights', 'freedom', 'liberty',
            'due process', 'equal protection', 'separation of powers', 'federalism',
            'supreme court', 'judicial review', 'constitutional', 'fundamental rights'
        ]
        
        criminal_keywords = [
            'criminal', 'defendant', 'prosecution', 'guilt', 'murder', 'theft',
            'fraud', 'assault', 'battery', 'sentencing', 'plea', 'trial',
            'evidence', 'witness', 'jury', 'verdict', 'penalty', 'punishment'
        ]
        
        contract_keywords = [
            'contract', 'agreement', 'breach', 'consideration', 'offer', 'acceptance',
            'damages', 'enforceable', 'terms', 'obligation', 'performance',
            'liability', 'indemnity', 'warranty', 'covenant', 'clause'
        ]
        
        # Check constitutional law first (highest priority)
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower 
               for keyword in constitutional_keywords):
            return 0
        
        # Check criminal law second
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower 
               for keyword in criminal_keywords):
            return 1
        
        # Check contract law third
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower 
               for keyword in contract_keywords):
            return 2
        
        return 3  # Default to "Other" category
    
    def _get_fallback_training_data(self):
        return {
            "texts": [
                "The First Amendment protects freedom of speech and religion",
                "Due process under the Fourteenth Amendment requires fair procedures",
                "The Constitution establishes the separation of powers",
                "Equal protection clause prohibits discrimination",
                "The Bill of Rights guarantees individual liberties",
                "Constitutional interpretation involves judicial review",
                "Federalism divides power between state and federal governments",
                "The Supreme Court has the power of judicial review",
                
                "The defendant is charged with first-degree murder",
                "Criminal intent is required for most crimes",
                "The prosecution must prove guilt beyond reasonable doubt",
                "Sentencing guidelines determine punishment for crimes",
                "Criminal procedure protects defendant rights",
                "The defendant has the right to remain silent",
                "Criminal law defines prohibited conduct",
                "Plea bargaining is common in criminal cases",
                
                "A valid contract requires offer, acceptance, and consideration",
                "Breach of contract occurs when terms are not fulfilled",
                "Contract law governs agreements between parties",
                "Damages may be awarded for contract breach",
                "A contract must have mutual assent to be enforceable",
                "Contract terms must be clear and definite",
                "Consideration is the exchange of value in contracts",
                "Contract interpretation follows objective standards",
                
                "Property law governs ownership rights",
                "Tort law addresses civil wrongs and damages",
                "Family law covers marriage and divorce",
                "Employment law regulates workplace relationships",
                "Tax law governs taxation and compliance",
                "Environmental law protects natural resources",
                "Intellectual property protects creative works",
                "Administrative law governs government agencies"
            ],
            "labels": [
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3
            ]
        }
    
    def train(self):
        try:
            training_data = self._create_training_data()
            
            X = self.vectorizer.fit_transform(training_data["texts"])
            y = np.array(training_data["labels"])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            
            y_pred = self.classifier.predict(X_test)
            logger.info("Classification Report:")
            logger.info(classification_report(y_test, y_pred, 
                                            target_names=list(self.categories.values())))
            
            self.is_trained = True
            logger.info("Legal text classifier trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
            raise
    
    def classify(self, text: str) -> Dict[str, Any]:
        if not self.is_trained:
            self.train()
        
        try:
            X = self.vectorizer.transform([text])
            
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            
            confidence = float(np.max(probabilities))
            
            category = self.categories[prediction]
            
            return {
                "category": category,
                "confidence": confidence,
                "probabilities": {
                    self.categories[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            }
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "category": "Other",
                "confidence": 0.0,
                "probabilities": {"Other": 1.0}
            }
    
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "categories": self.categories,
            "is_trained": self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data["vectorizer"]
            self.classifier = model_data["classifier"]
            self.categories = model_data["categories"]
            self.is_trained = model_data["is_trained"]
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

legal_classifier = LegalTextClassifier()
