import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
import os
import time
import json
import requests
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv()

# Оптимизация для M1 Max
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Импортируем базовые классы из основного файла
from transformer import (
    PositionalEncoding, MultiHeadAttention, FeedForward, 
    EncoderLayer, DecoderLayer, TimeSeriesTransformer,
    TimeSeriesDataset, create_synthetic_data, visualize_results, forecast
)

@dataclass
class TrainingMetrics:
    """Метрики для анализа LLM"""
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    loss_improvement: float
    gradient_norm: float
    training_time: float
    overfitting_score: float = 0.0
    convergence_score: float = 0.0
    
    def __post_init__(self):
        # Новый расчет overfitting_score: знак сохраняется
        gap = self.train_loss - self.val_loss
        self.overfitting_score = gap / (abs(self.val_loss) + 1e-8)
        self.convergence_score = abs(self.loss_improvement) / max(abs(self.val_loss), 1e-8)

class OptimizationAction(Enum):
    """Возможные действия оптимизатора"""
    CONTINUE = "continue"
    ADJUST_LR = "adjust_lr"
    ADJUST_DROPOUT = "adjust_dropout"
    EARLY_STOP = "early_stop"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    INCREASE_BATCH_SIZE = "increase_batch_size"
    ENABLE_GRADIENT_CLIPPING = "enable_gradient_clipping"
    ADJUST_WEIGHT_DECAY = "adjust_weight_decay"

@dataclass
class OptimizationSuggestion:
    """Рекомендация от оптимизатора"""
    action: OptimizationAction
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    priority: int = 1

class LLMHyperparameterOptimizer:
    """Интеллектуальный оптимизатор гиперпараметров с LLM"""
    
    def __init__(self, model_type: str = "rule_based", api_key: Optional[str] = None):
        self.model_type = model_type
        self.api_key = api_key
        self.history: List[TrainingMetrics] = []
        self.suggestions_history: List[OptimizationSuggestion] = []
        self.performance_tracker = {}
        
        # Настройки для rule-based оптимизации
        self.rules_config = {
            'overfitting_threshold': 0.15,
            'stagnation_patience': 5,
            'lr_reduction_factor': 0.5,
            'dropout_increase_step': 0.05,
            'gradient_clip_threshold': 1.0
        }
        
        print(f"🧠 Инициализирован LLM оптимизатор: {model_type}")
    
    def analyze_training_progress(self, metrics: TrainingMetrics) -> OptimizationSuggestion:
        """Анализ прогресса обучения и генерация рекомендаций"""
        self.history.append(metrics)
        
        if self.model_type == "rule_based":
            return self._rule_based_analysis(metrics)
        elif self.model_type == "local_llm":
            return self._local_llm_analysis(metrics)
        elif self.model_type == "openai":
            return self._openai_analysis(metrics)
        else:
            return OptimizationSuggestion(
                action=OptimizationAction.CONTINUE,
                parameters={},
                confidence=0.5,
                reasoning="Unknown optimizer type"
            )
    
    def _rule_based_analysis(self, metrics: TrainingMetrics) -> OptimizationSuggestion:
        """Rule-based анализ (быстрый и надежный)"""
        # 1. ИСПРАВЛЕНИЕ БАГА #2: Проверка на переобучение (train_loss < val_loss означает overfitting)
        gap = metrics.train_loss - metrics.val_loss
        if gap < 0 and abs(metrics.overfitting_score) > self.rules_config['overfitting_threshold']:
            if len(self.history) >= 3:
                recent = self.history[-3:]
                if all((m.train_loss - m.val_loss) < 0 and abs(m.overfitting_score) > self.rules_config['overfitting_threshold'] for m in recent):
                    return OptimizationSuggestion(
                        action=OptimizationAction.ADJUST_DROPOUT,
                        parameters={'dropout_increase': self.rules_config['dropout_increase_step']},
                        confidence=0.8,
                        reasoning=f"Переобучение обнаружено (train < val, score: {metrics.overfitting_score:.3f}). Увеличиваем dropout.",
                        priority=1
                    )
        
        # 2. Проверка стагнации
        if len(self.history) >= self.rules_config['stagnation_patience']:
            recent_improvements = [m.loss_improvement for m in self.history[-self.rules_config['stagnation_patience']:]]
            if all(abs(imp) < 0.001 for imp in recent_improvements):
                return OptimizationSuggestion(
                    action=OptimizationAction.ADJUST_LR,
                    parameters={'lr_factor': self.rules_config['lr_reduction_factor']},
                    confidence=0.7,
                    reasoning=f"Стагнация обучения. Уменьшаем learning rate в {1/self.rules_config['lr_reduction_factor']:.1f} раз.",
                    priority=2
                )
        
        # 3. Проверка градиентов
        if metrics.gradient_norm > 10.0:
            return OptimizationSuggestion(
                action=OptimizationAction.ENABLE_GRADIENT_CLIPPING,
                parameters={'max_norm': self.rules_config['gradient_clip_threshold']},
                confidence=0.9,
                reasoning=f"Большие градиенты (norm: {metrics.gradient_norm:.2f}). Включаем gradient clipping.",
                priority=1
            )
        
        # 4. Проверка скорости сходимости
        if len(self.history) >= 10:
            recent_losses = [m.val_loss for m in self.history[-10:]]
            if recent_losses[0] - recent_losses[-1] < 0.001:  # Очень медленная сходимость
                return OptimizationSuggestion(
                    action=OptimizationAction.ADJUST_LR,
                    parameters={'lr_factor': 1.2},  # Небольшое увеличение
                    confidence=0.6,
                    reasoning="Медленная сходимость. Немного увеличиваем learning rate.",
                    priority=3
                )
        
        # 5. Все хорошо - продолжаем
        return OptimizationSuggestion(
            action=OptimizationAction.CONTINUE,
            parameters={},
            confidence=0.9,
            reasoning="Обучение идет стабильно. Продолжаем без изменений.",
            priority=5
        )
    
    def _local_llm_analysis(self, metrics: TrainingMetrics) -> OptimizationSuggestion:
        """Анализ с помощью локальной LLM через Ollama"""
        try:
            prompt = self._create_analysis_prompt(metrics)
            
            # Запрос к Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen2.5:latest',  # Используем qwen2.5 вместо qwen3
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,
                        'top_p': 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '')
                
                # Парсим JSON ответ от LLM
                try:
                    # Ищем JSON в ответе
                    import re
                    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                    if json_match:
                        suggestion_data = json.loads(json_match.group())
                        
                        action_map = {
                            'continue': OptimizationAction.CONTINUE,
                            'adjust_lr': OptimizationAction.ADJUST_LR,
                            'adjust_dropout': OptimizationAction.ADJUST_DROPOUT,
                            'early_stop': OptimizationAction.EARLY_STOP,
                            'enable_gradient_clipping': OptimizationAction.ENABLE_GRADIENT_CLIPPING,
                            'adjust_weight_decay': OptimizationAction.ADJUST_WEIGHT_DECAY
                        }
                        
                        action = action_map.get(suggestion_data.get('action', 'continue'), OptimizationAction.CONTINUE)
                        
                        return OptimizationSuggestion(
                            action=action,
                            parameters=suggestion_data.get('parameters', {}),
                            confidence=suggestion_data.get('confidence', 0.7),
                            reasoning=f"🤖 Ollama: {suggestion_data.get('reasoning', 'LLM рекомендация')}"
                        )
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️ Ошибка парсинга ответа Ollama: {e}")
                    print(f"Ответ: {llm_response[:200]}...")
            else:
                print(f"⚠️ Ошибка Ollama API: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Ollama недоступен: {e}")
        except Exception as e:
            print(f"⚠️ Ошибка локальной LLM: {e}")
        
        # Fallback на rule-based анализ
        print("🔄 Используем rule-based анализ как fallback")
        return self._rule_based_analysis(metrics)
    
    def _openai_analysis(self, metrics: TrainingMetrics) -> OptimizationSuggestion:
        """Анализ с помощью OpenAI API (structured output через function calling)"""
        try:
            from openai import OpenAI
            
            # Получаем API ключ из переменных окружения
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key or api_key == 'your_api_key_here':
                print("⚠️ OPENAI_API_KEY не найден или не настроен в .env файле")
                print("   Пожалуйста, добавьте ваш реальный API ключ в .env файл")
                return self._rule_based_analysis(metrics)
            
            print(f"🔑 Используем OpenAI API (ключ: {api_key[:8]}...)")
            
            # Создаем клиент OpenAI
            client = OpenAI(api_key=api_key)
            
            prompt = self._create_analysis_prompt(metrics)
            
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert ML engineer specializing in hyperparameter optimization. Analyze the metrics and call the optimize_hyperparameters function with your recommendation."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "optimize_hyperparameters",
                            "description": "Provide hyperparameter optimization recommendation",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["continue", "adjust_lr", "adjust_dropout", "early_stop", "enable_gradient_clipping", "adjust_weight_decay"],
                                        "description": "The optimization action to take"
                                    },
                                    "factor": {
                                        "type": "number",
                                        "description": "Learning rate adjustment factor (for adjust_lr action)"
                                    },
                                    "dropout_increase": {
                                        "type": "number",
                                        "description": "Dropout increase amount (for adjust_dropout action)"
                                    },
                                    "max_norm": {
                                        "type": "number", 
                                        "description": "Max gradient norm (for enable_gradient_clipping action)"
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Confidence in the recommendation (0-1)"
                                    },
                                    "reasoning": {
                                        "type": "string",
                                        "description": "Detailed explanation for the recommendation"
                                    }
                                },
                                "required": ["action", "confidence", "reasoning"]
                            }
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "optimize_hyperparameters"}},
                temperature=0.3
            )
            
            # Получаем структурированный ответ через tool call
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                print("⚠️ OpenAI не вернул tool call")
                return self._rule_based_analysis(metrics)
            
            # Парсим аргументы функции
            suggestion_data = json.loads(tool_calls[0].function.arguments)
            
            action_map = {
                'continue': OptimizationAction.CONTINUE,
                'adjust_lr': OptimizationAction.ADJUST_LR,
                'adjust_dropout': OptimizationAction.ADJUST_DROPOUT,
                'early_stop': OptimizationAction.EARLY_STOP,
                'enable_gradient_clipping': OptimizationAction.ENABLE_GRADIENT_CLIPPING,
                'adjust_weight_decay': OptimizationAction.ADJUST_WEIGHT_DECAY
            }
            
            action = action_map.get(suggestion_data.get('action', 'continue'), OptimizationAction.CONTINUE)
            
            # Собираем параметры в нужном формате
            parameters = {}
            if 'factor' in suggestion_data:
                parameters['factor'] = suggestion_data['factor']
                parameters['lr_factor'] = suggestion_data['factor']  # для совместимости
            if 'dropout_increase' in suggestion_data:
                parameters['dropout_increase'] = suggestion_data['dropout_increase']
                parameters['amount'] = suggestion_data['dropout_increase']  # для совместимости
            if 'max_norm' in suggestion_data:
                parameters['max_norm'] = suggestion_data['max_norm']
                parameters['clip_norm'] = suggestion_data['max_norm']  # для совместимости
                
            suggestion = OptimizationSuggestion(
                    action=action,
                    parameters=parameters,
                    confidence=suggestion_data.get('confidence', 0.8),
                    reasoning=suggestion_data.get('reasoning', '')  # Пустая строка вместо длинного текста
                )
            print(f"✅ GPT-4 предложил: {action.value} с параметрами {parameters}")
            return suggestion
                
        except ImportError:
            print("⚠️ Библиотека openai не установлена. Установите: pip install openai")
        except Exception as e:
            print(f"⚠️ Ошибка OpenAI API: {e}")
            import traceback
            traceback.print_exc()
        
        # Fallback на rule-based анализ
        print("🔄 Используем rule-based анализ как fallback")
        return self._rule_based_analysis(metrics)
    
    def _create_analysis_prompt(self, metrics: TrainingMetrics) -> str:
        """Создание промпта для LLM анализа"""
        history_summary = ""
        if len(self.history) > 1:
            recent_history = self.history[-5:]  # Последние 5 эпох
            history_summary = "\nRecent training history:\n"
            for i, h in enumerate(recent_history):
                history_summary += f"Epoch {h.epoch}: train_loss={h.train_loss:.4f}, val_loss={h.val_loss:.4f}, lr={h.learning_rate:.6f}\n"
        
        prompt = f"""
You are an expert ML engineer optimizing a Transformer model for time series forecasting.

Current training metrics:
- Epoch: {metrics.epoch}
- Train Loss: {metrics.train_loss:.4f}
- Validation Loss: {metrics.val_loss:.4f}
- Learning Rate: {metrics.learning_rate:.6f}
- Loss Improvement: {metrics.loss_improvement:.4f}
- Gradient Norm: {metrics.gradient_norm:.2f}
- Training Time: {metrics.training_time:.2f}s
- Overfitting Score: {metrics.overfitting_score:.3f}
- Convergence Score: {metrics.convergence_score:.3f}

{history_summary}

Model details:
- Architecture: Transformer (encoder-decoder)
- Task: Time series forecasting
- Data: Synthetic time series with trend and seasonality
- Device: Apple M1 Max with MPS acceleration

Please analyze the training progress and suggest ONE specific action:
1. continue - if training is going well
2. adjust_lr - if learning rate needs adjustment (specify factor)
3. adjust_dropout - if overfitting detected (specify increase amount)
4. early_stop - if training should stop
5. enable_gradient_clipping - if gradients are exploding
6. adjust_weight_decay - if regularization needs adjustment

Respond in JSON format:
{{
    "action": "action_name",
    "parameters": {{"param": value}},
    "confidence": 0.8,
    "reasoning": "Detailed explanation"
}}
"""
        return prompt
    
    def apply_suggestions(self, suggestion: OptimizationSuggestion, optimizer: optim.Optimizer, 
                         model: nn.Module, gradient_clipping_state: Dict[str, Any]) -> bool:
        """Применение рекомендаций оптимизатора"""
        self.suggestions_history.append(suggestion)
        try:
            if suggestion.action == OptimizationAction.ADJUST_LR:
                factor = suggestion.parameters.get('lr_factor') or suggestion.parameters.get('factor') or 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= factor
                # Улучшенное форматирование для factor
                if factor > 1.0:
                    print(f"🔧 LLM: Увеличен learning rate на {factor:.2f}x ↑ (новый LR: {optimizer.param_groups[0]['lr']:.6f})")
                else:
                    print(f"🔧 LLM: Уменьшен learning rate на {factor:.2f}x ↓ (новый LR: {optimizer.param_groups[0]['lr']:.6f})")
                return True
            elif suggestion.action == OptimizationAction.ADJUST_DROPOUT:
                dropout_increase = suggestion.parameters.get('dropout_increase') or suggestion.parameters.get('amount') or 0.05
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = min(0.5, module.p + dropout_increase)
                print(f"🔧 LLM: Увеличен dropout на {dropout_increase:.2f}")
                return True
            elif suggestion.action == OptimizationAction.ENABLE_GRADIENT_CLIPPING:
                max_norm = suggestion.parameters.get('max_norm') or suggestion.parameters.get('clip_norm') or 1.0
                # ИСПРАВЛЕНИЕ БАГА #1: Реально включаем gradient clipping
                gradient_clipping_state['enabled'] = True
                gradient_clipping_state['max_norm'] = max_norm
                print(f"🔧 LLM: Включен gradient clipping с max_norm={max_norm}")
                return True
            elif suggestion.action == OptimizationAction.EARLY_STOP:
                print("🛑 LLM: Рекомендует early stopping")
                return False
            elif suggestion.action == OptimizationAction.CONTINUE:
                print(f"✅ LLM: продолжаем обучение")
                return True
        except Exception as e:
            print(f"❌ Ошибка применения рекомендации: {e}")
        return True
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Сводка производительности оптимизатора"""
        if not self.history:
            return {}
        
        best_val_loss = min(m.val_loss for m in self.history)
        best_epoch = next(i for i, m in enumerate(self.history) if m.val_loss == best_val_loss)
        
        actions_count = {}
        for suggestion in self.suggestions_history:
            action = suggestion.action.value
            actions_count[action] = actions_count.get(action, 0) + 1
        
        return {
            'total_epochs': len(self.history),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'final_val_loss': self.history[-1].val_loss if self.history else 0,
            'improvement': (self.history[0].val_loss - best_val_loss) / self.history[0].val_loss if self.history else 0,
            'actions_taken': actions_count,
            'avg_confidence': np.mean([s.confidence for s in self.suggestions_history]) if self.suggestions_history else 0
        }
    
    def visualize_optimization_history(self):
        """Визуализация истории оптимизации"""
        if len(self.history) < 2:
            print("Недостаточно данных для визуализации")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = [m.epoch for m in self.history]
        train_losses = [m.train_loss for m in self.history]
        val_losses = [m.val_loss for m in self.history]
        learning_rates = [m.learning_rate for m in self.history]
        overfitting_scores = [m.overfitting_score for m in self.history]
        
        # График потерь
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', alpha=0.7)
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress with LLM Optimization')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График learning rate
        axes[0, 1].plot(epochs, learning_rates, color='orange', marker='o', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('LLM Learning Rate Adjustments')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # График переобучения
        axes[1, 0].plot(epochs, overfitting_scores, color='red', alpha=0.7)
        axes[1, 0].axhline(y=self.rules_config['overfitting_threshold'], color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Overfitting Score')
        axes[1, 0].set_title('Overfitting Detection')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Действия оптимизатора
        actions_count = {}
        for suggestion in self.suggestions_history:
            action = suggestion.action.value
            actions_count[action] = actions_count.get(action, 0) + 1
        
        if actions_count:
            actions = list(actions_count.keys())
            counts = list(actions_count.values())
            axes[1, 1].bar(actions, counts, alpha=0.7)
            axes[1, 1].set_xlabel('Action')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('LLM Actions Taken')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def train_model_with_llm(model: nn.Module, 
                        train_loader: DataLoader, 
                        val_loader: DataLoader,
                        n_epochs: int = 100,
                        learning_rate: float = 0.001,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        llm_optimizer_type: str = "rule_based",
                        llm_every_n_epochs: int = 3) -> Tuple[List[float], List[float], LLMHyperparameterOptimizer]:
    """Обучение модели с LLM оптимизацией (с исправленными багами)"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    llm_optimizer = LLMHyperparameterOptimizer(model_type=llm_optimizer_type)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    best_model_state = None
    
    # ИСПРАВЛЕНИЕ БАГА #1: Состояние gradient clipping в словаре
    gradient_clipping_state = {'enabled': False, 'max_norm': 1.0}
    
    print(f"\n🚀 Начинаем обучение с LLM оптимизацией ({llm_optimizer_type})")
    print(f"   🛑 Early stopping: patience={patience}")
    training_start_time = time.time()
    no_improve_epochs = 0
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        total_grad_norm = 0
        train_start_time = time.time()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            optimizer.zero_grad()
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(output, tgt)
            loss.backward()
            
            # ИСПРАВЛЕНИЕ БАГА #1: Правильный gradient clipping
            if gradient_clipping_state['enabled']:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_state['max_norm'])
            else:
                # Вычисляем норму без клиппинга для логирования
                grad_norm = torch.norm(torch.stack([
                    p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None
                ]), 2).item()
            
            total_grad_norm += grad_norm
            optimizer.step()
            train_loss += loss.item()
        train_time = time.time() - train_start_time
        avg_grad_norm = total_grad_norm / len(train_loader)
        model.eval()
        val_loss = 0
        val_start_time = time.time()
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = torch.cat([src[:, -1:, :], tgt[:, :-1, :]], dim=1)
                tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                output = model(src, tgt_input, tgt_mask=tgt_mask)
                loss = criterion(output, tgt)
                val_loss += loss.item()
        val_time = time.time() - val_start_time
        epoch_time = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        loss_improvement = (val_losses[-2] - avg_val_loss) if len(val_losses) > 1 else 0.0
        metrics = TrainingMetrics(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            learning_rate=optimizer.param_groups[0]['lr'],
            loss_improvement=loss_improvement,
            gradient_norm=avg_grad_norm,
            training_time=epoch_time
        )
        force_trigger = no_improve_epochs >= 5
        if epoch % llm_every_n_epochs == 0 or force_trigger:
            suggestion = llm_optimizer.analyze_training_progress(metrics)
            should_continue = llm_optimizer.apply_suggestions(suggestion, optimizer, model, gradient_clipping_state)
            print(f"[LLM] epoch={epoch} action={suggestion.action.value} params={suggestion.parameters} val_loss={avg_val_loss:.4f}")
        else:
            should_continue = True
        
        # ИСПРАВЛЕНИЕ БАГА #3: Правильная логика no_improve_epochs
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0  # Сбрасываем только здесь
        else:
            patience_counter += 1
            no_improve_epochs += 1
        if epoch % 5 == 0:
            print(f'Epoch {epoch:3d}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
            print(f'   ⏱️  Train: {train_time:.2f}s | Val: {val_time:.2f}s | Total: {epoch_time:.2f}s | Speed: {len(train_loader)/train_time:.1f} batch/s')
            print(f'   🎯 Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}')
        if not should_continue or patience_counter >= patience:
            if not should_continue:
                print(f'\n🛑 LLM рекомендует остановить обучение на эпохе {epoch}!')
            else:
                print(f'\n🛑 Early stopping на эпохе {epoch}! Val loss не улучшался {patience} эпох.')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Загружена лучшая модель с Val Loss: {best_val_loss:.6f}")
    total_training_time = time.time() - training_start_time
    print(f"\n🎯 Обучение завершено!")
    print(f"   Общее время: {total_training_time:.1f}s ({total_training_time/60:.1f} мин)")
    print(f"   Время на эпоху: {total_training_time/len(train_losses):.1f}s")
    print(f"   Финальный Train Loss: {train_losses[-1]:.6f}")
    print(f"   Лучший Val Loss: {best_val_loss:.6f}")
    performance = llm_optimizer.get_performance_summary()
    print(f"\n🧠 LLM Оптимизация:")
    print(f"   Улучшение: {performance.get('improvement', 0)*100:.1f}%")
    print(f"   Действий выполнено: {sum(performance.get('actions_taken', {}).values())}")
    print(f"   Средняя уверенность: {performance.get('avg_confidence', 0):.2f}")
    actions_count = performance.get('actions_taken', {})
    if actions_count:
        print("   Топ-5 действий LLM:")
        for i, (action, count) in enumerate(sorted(actions_count.items(), key=lambda x: -x[1])[:5]):
            print(f"      {i+1}. {action}: {count} раз")
    return train_losses, val_losses, llm_optimizer

def main():
    """Основная функция с LLM оптимизацией"""
    script_start_time = time.time()
    
    # Параметры
    seq_length = 60
    pred_length = 15
    batch_size = 32
    n_epochs = 30   # Меньше эпох для тестирования OpenAI
    
    # Создание данных
    print("📊 Создание синтетических данных...")
    data_start_time = time.time()
    data = create_synthetic_data(2000)
    data_prep_time = time.time() - data_start_time
    print(f"   ✅ Данные созданы за {data_prep_time:.2f}s")
    
    # Разделение и нормализация
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    mean = train_data.mean()
    std = train_data.std()
    train_data_norm = (train_data - mean) / std
    test_data_norm = (test_data - mean) / std
    
    # Создание датасетов
    train_dataset = TimeSeriesDataset(train_data_norm[:-100], seq_length, pred_length, augment=False)
    val_dataset = TimeSeriesDataset(train_data_norm[-100-seq_length-pred_length:], seq_length, pred_length, augment=False)
    
    # DataLoader для MPS: num_workers=0, pin_memory=False, persistent_workers=False
    if torch.backends.mps.is_available():
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, persistent_workers=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    # Определяем устройство
    if torch.backends.mps.is_available():
        device = 'mps'
        print("🚀 Используем Metal Performance Shaders (MPS) для ускорения!")
        print(f"   MPS built: {torch.backends.mps.is_built()}")
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = 'cuda'
        print("Используем CUDA для ускорения!")
    else:
        device = 'cpu'
        print("Используем CPU (медленно)")
    
    # Создание модели
    print("\n🧠 Создание модели трансформера...")
    model_start_time = time.time()
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=128,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        d_ff=512,
        max_seq_length=200,
        dropout=0.2
    )
    
    model_creation_time = time.time() - model_start_time
    print(f"   ✅ Модель создана за {model_creation_time:.2f}s")
    print(f"   📊 Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   🖥️  Устройство: {device}")
    print(f"   📈 Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"   ⚙️  Batch size: {batch_size} | Epochs: {n_epochs}")
    
    # Обучение с LLM оптимизацией
    print("\n🤖 Обучение модели с LLM оптимизацией...")
    train_phase_start = time.time()
    train_losses, val_losses, llm_optimizer = train_model_with_llm(
        model, train_loader, val_loader, 
        n_epochs=n_epochs, 
        learning_rate=0.001, 
        device=device,
        llm_optimizer_type="openai",  # Тестируем с OpenAI!
        llm_every_n_epochs=3
    )
    train_phase_time = time.time() - train_phase_start
    
    # Прогнозирование на тестовых данных
    print("\n🔮 Выполнение прогнозирования...")
    print(f"   📏 Длина тестовых данных: {len(test_data_norm)}")
    prediction_start_time = time.time()
    model.eval()
    test_predictions = []
    
    # Используем скользящее окно для прогнозирования
    prediction_steps = 0
    for i in range(0, len(test_data_norm) - seq_length, pred_length):
        src = torch.FloatTensor(test_data_norm[i:i+seq_length]).unsqueeze(0)
        pred = forecast(model, src, pred_length, device, temperature=1.0)
        test_predictions.extend(pred.squeeze().cpu().numpy())
        prediction_steps += 1
        
        if prediction_steps % 5 == 0:
            print(f"   🔄 Выполнено {prediction_steps} шагов прогнозирования...")
        
        if i + seq_length + pred_length >= len(test_data_norm):
            break
    
    prediction_time = time.time() - prediction_start_time
    print(f"   ✅ Прогнозирование завершено за {prediction_time:.2f}s")
    print(f"   📊 Всего шагов: {prediction_steps} | Предсказаний: {len(test_predictions)}")
    print(f"   ⚡ Скорость: {prediction_steps/prediction_time:.1f} шагов/сек")
    
    # Денормализация предсказаний
    test_predictions = np.array(test_predictions) * std + mean
    
    # Визуализация результатов прогнозирования
    print("\n📊 Визуализация результатов прогнозирования...")
    viz_start_time = time.time()
    visualize_results(
        train_data.squeeze(), 
        test_data[:len(test_predictions)].squeeze(),
        test_predictions.squeeze(),
        train_losses,
        val_losses
    )
    viz_time = time.time() - viz_start_time
    print(f"   ✅ Графики прогнозов построены за {viz_time:.2f}s")
    
    # Расчет метрик
    actual = test_data[:len(test_predictions)].squeeze()
    predicted = test_predictions.squeeze()
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    
    print(f"\n📈 Метрики прогнозирования:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {np.sqrt(mse):.4f}")
    
    # Визуализация LLM оптимизации
    print("\n📊 Визуализация LLM оптимизации...")
    llm_optimizer.visualize_optimization_history()
    
    # Сохранение результатов
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_predictions': test_predictions.tolist() if isinstance(test_predictions, np.ndarray) else test_predictions,
        'metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse))
        },
        'llm_performance': llm_optimizer.get_performance_summary(),
        'model_config': {
            'input_dim': 1,
            'd_model': 128,
            'n_heads': 8,
            'n_encoder_layers': 4,
            'n_decoder_layers': 4,
            'd_ff': 512,
            'max_seq_length': 200,
            'dropout': 0.2
        }
    }
    
    # Сохранение модели
    save_start_time = time.time()
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std,
        'model_config': results['model_config'],
        'llm_performance': results['llm_performance']
    }, 'llm_transformer_model.pth')
    save_time = time.time() - save_start_time
    
    with open('llm_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Модель сохранена за {save_time:.2f}s в 'llm_transformer_model.pth'")
    print(f"💾 Результаты сохранены в 'llm_optimization_results.json'")
    
    total_script_time = time.time() - script_start_time
    print(f"\n🎯 ОБЩЕЕ ВРЕМЯ ВЫПОЛНЕНИЯ: {total_script_time:.1f}s ({total_script_time/60:.1f} мин)")
    print(f"   📊 Подготовка данных: {data_prep_time:.2f}s")
    print(f"   🧠 Создание модели: {model_creation_time:.2f}s") 
    print(f"   🤖 LLM обучение: {train_phase_time:.1f}s")
    print(f"   🔮 Прогнозирование: {prediction_time:.2f}s")
    print(f"   📊 Визуализация: {viz_time:.2f}s")

if __name__ == "__main__":
    main()