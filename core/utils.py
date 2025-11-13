"""
Общи помощни функции за AI Football Prediction Service
"""

import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np


def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Настройка на logging система
    
    Args:
        log_dir: Директория за log файлове
        log_level: Ниво на логване (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Logger обект
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"football_ai_{datetime.now().strftime('%Y%m%d')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("FootballAI")
    logger.info("Logging система инициализирана")
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Зареждане на YAML конфигурация
    
    Args:
        config_path: Път до YAML файл
    
    Returns:
        Dictionary с конфигурация
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Запазване на JSON файл
    
    Args:
        data: Dictionary за запазване
        filepath: Път до файл
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Зареждане на JSON файл
    
    Args:
        filepath: Път до файл
    
    Returns:
        Dictionary с данни
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid функция
    
    Args:
        x: Input array
    
    Returns:
        Sigmoid трансформация
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def soft_clip_probs(p, eps=1e-6, lo=0.02, hi=0.98):
    """
    Soft clipping на вероятности с temperature-like squeeze
    
    Args:
        p: Вероятности (array или float)
        eps: Минимална граница за clipping
        lo: Долна soft граница
        hi: Горна soft граница
    
    Returns:
        Soft-clipped вероятности
    """
    p = np.clip(p, eps, 1-eps)
    # Temperature-like squeeze around 0.5
    return lo + (hi-lo) * (p - eps) / (1 - 2*eps)


def normalize_1x2_probs(probs):
    """
    Нормализира 1X2 вероятности към сума 1
    
    Args:
        probs: Array с 3 вероятности [prob_1, prob_X, prob_2]
    
    Returns:
        Нормализирани вероятности
    """
    probs = np.array(probs)
    probs = soft_clip_probs(probs)
    return probs / probs.sum()


def normalize_feature(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Нормализация на feature
    
    Args:
        series: Pandas Series
        method: 'minmax' или 'zscore'
    
    Returns:
        Нормализиран Series
    """
    if method == 'minmax':
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0, index=series.index)
        return (series - mean) / std
    
    else:
        raise ValueError(f"Неизвестен метод за нормализация: {method}")


def calculate_rolling_average(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1
) -> pd.Series:
    """
    Изчисляване на rolling average за дадена група
    
    Args:
        df: DataFrame
        group_col: Колона за групиране
        value_col: Колона със стойности
        window: Размер на прозореца
        min_periods: Минимален брой наблюдения
    
    Returns:
        Series с rolling averages
    """
    return df.groupby(group_col)[value_col].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).mean()
    )


def get_project_root() -> Path:
    """
    Връща root директорията на проекта
    
    Returns:
        Path обект
    """
    return Path(__file__).parent.parent


def safe_divide(numerator, denominator, default: float = 0.0):
    """
    Безопасно деление (избягва division by zero)
    Работи както с единични стойности, така и с pandas Series
    
    Args:
        numerator: Числител (float или Series)
        denominator: Знаменател (float или Series)
        default: Стойност при деление на 0
    
    Returns:
        Резултат от делението (float или Series)
    """
    # Ако работим с pandas Series
    if isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(default)
    
    # Ако работим с единични стойности
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def date_to_datetime(date_str: str) -> datetime:
    """
    Конвертиране на string към datetime
    
    Args:
        date_str: Дата като string
    
    Returns:
        datetime обект
    """
    try:
        return pd.to_datetime(date_str)
    except:
        return None


def calculate_days_between(date1: datetime, date2: datetime) -> int:
    """
    Изчисляване на дни между две дати
    
    Args:
        date1: Първа дата
        date2: Втора дата
    
    Returns:
        Брой дни
    """
    if pd.isna(date1) or pd.isna(date2):
        return 0
    return abs((date2 - date1).days)


class PerformanceTimer:
    """Context manager за измерване на време за изпълнение"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger("FootballAI")
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Стартиране на: {self.name}")
        return self
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"Завършено: {self.name} за {elapsed:.2f} секунди")
