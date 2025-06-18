# src/trackie/config/settings.py

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Importa as ferramentas certas: BaseModel para estrutura, BaseSettings para ler envs
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings as PydanticBaseSettings

from ..utils.logger import get_logger

logger = get_logger(__name__)

# --- Base Settings para ler variáveis de ambiente ---
# Criamos uma classe base para que Pydantic saiba como procurar por .env files.
class EnvBaseSettings(PydanticBaseSettings):
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

# --- Modelos de Dados Pydantic para Validação ---

class UserSettings(BaseModel):
    name: str = "Usuário"

# Gemini e OpenAI herdam de EnvBaseSettings para ler a API_KEY do ambiente
class GeminiSettings(EnvBaseSettings):
    api_key: str = Field(..., alias="GEMINI_API_KEY") # Pega a chave da env GEMINI_API_KEY
    model: str = "models/gemini-2.0-flash-live-001"
    temperature: float = 0.2

class OpenAISettings(EnvBaseSettings):
    api_key: str = Field(..., alias="OPENAI_API_KEY") # Pega a chave da env OPENAI_API_KEY
    model: str = "gpt-4o"
    temperature: float = 0.7

class LLMSettings(BaseModel):
    provider: str
    system_prompt_path: Path
    
    # As seções do provedor são opcionais no início
    gemini: Optional[GeminiSettings] = None
    openai: Optional[OpenAISettings] = None

    @model_validator(mode='after')
    def check_provider_config(self) -> 'LLMSettings':
        """Valida se a configuração para o provedor ativo existe e está completa."""
        provider = self.provider.lower()
        
        if provider == 'gemini':
            if not self.gemini:
                raise ValueError("Provedor é 'gemini', mas a seção 'gemini:' está faltando no settings.yml.")
            if not self.gemini.api_key:
                 raise ValueError("Provedor é 'gemini', mas a variável de ambiente 'GEMINI_API_KEY' não foi encontrada.")
        
        elif provider == 'openai':
            if not self.openai:
                raise ValueError("Provedor é 'openai', mas a seção 'openai:' está faltando no settings.yml.")
            if not self.openai.api_key:
                 raise ValueError("Provedor é 'openai', mas a variável de ambiente 'OPENAI_API_KEY' não foi encontrada.")
        
        else:
            raise ValueError(f"Provedor de LLM desconhecido: '{self.provider}'. Use 'gemini' ou 'openai'.")
            
        return self

class VideoSettings(BaseModel):
    provider: str = "opencv"
    device_index: int = 0
    fps: float = 1.0
    jpeg_quality: int = Field(50, ge=10, le=100)

class AudioSettings(BaseModel):
    chunk_size: int = 1024
    send_sample_rate: int = 16000
    receive_sample_rate: int = 24000
    channels: int = 1

class VisionSettings(BaseModel):
    yolo_model_path: Path
    confidence_threshold: float = Field(0.45, ge=0.1, le=1.0)
    midas_model_path: Path
    deepface_db_path: Path
    deepface_model_name: str
    deepface_detector_backend: str
    deepface_distance_metric: str

class PathSettings(BaseModel):
    data: Path
    tool_definitions: Path
    danger_sound: Path

class AppSettings(BaseModel):
    """O modelo Pydantic principal que engloba todas as configurações."""
    user: UserSettings
    llm: LLMSettings
    video: VideoSettings
    audio: AudioSettings
    vision: VisionSettings
    paths: PathSettings


def load_settings(config_path: Path = Path("config/settings.yml")) -> AppSettings:
    """
    Carrega, valida e retorna as configurações da aplicação.
    """
    # Verifica se o caminho já é absoluto. Se não, constrói a partir da raiz do projeto.
    if not config_path.is_absolute():
        # AQUI ESTÁ A CORREÇÃO: mudamos de .parents[2] para .parents[3]
        # Isso sobe de:
        # settings.py -> .../config -> .../trackie -> .../src -> C:\TrackWay (Raiz)
        project_root = Path(__file__).resolve().parents[3]
        config_path = project_root / config_path

    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado em: {config_path}")
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    logger.info(f"Carregando configurações de: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        settings = AppSettings(**config_data)
        
        logger.info("Configurações carregadas e validadas com sucesso.")
        return settings

    except yaml.YAMLError as e:
        logger.critical(f"Erro de sintaxe no arquivo de configuração YAML: {e}")
        raise
    except ValidationError as e:
        logger.critical(f"Erro de validação nas configurações: {e}")
        raise
    except Exception as e:
        logger.critical(f"Erro inesperado ao carregar as configurações: {e}")
        raise

# --- Instância Global ---
try:
    settings = load_settings()
except (FileNotFoundError, ValueError) as e:
    logger.critical(f"Não foi possível carregar as configurações. Encerrando. Erro: {e}")
    settings = None