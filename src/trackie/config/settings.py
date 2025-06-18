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


# --- Função Auxiliar para Encontrar a Raiz do Projeto ---
def find_project_root(marker_file: str = "pyproject.toml") -> Path:
    """
    Encontra a raiz do projeto de forma robusta, procurando por um arquivo marcador.

    Args:
        marker_file: O nome do arquivo que indica a raiz (ex: 'pyproject.toml' ou '.git').

    Returns:
        O caminho (Path) para o diretório raiz do projeto.
    
    Raises:
        FileNotFoundError: Se o arquivo marcador não for encontrado.
    """
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / marker_file).exists():
            logger.debug(f"Raiz do projeto encontrada em: {parent} (marcador: {marker_file})")
            return parent
    raise FileNotFoundError(f"Não foi possível encontrar a raiz do projeto. Arquivo marcador '{marker_file}' não localizado.")


# --- Base Settings para ler variáveis de ambiente ---
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
    api_key: str = Field(..., alias="GEMINI_API_KEY")
    model: str = "models/gemini-1.5-flash-latest"
    temperature: float = 0.2

class OpenAISettings(EnvBaseSettings):
    api_key: str = Field(..., alias="OPENAI_API_KEY")
    model: str = "gpt-4o"
    temperature: float = 0.7

class LLMSettings(BaseModel):
    provider: str
    system_prompt_path: Path
    gemini: Optional[GeminiSettings] = None
    openai: Optional[OpenAISettings] = None

    @model_validator(mode='after')
    def check_provider_config(self) -> 'LLMSettings':
        """Valida se a configuração para o provedor ativo existe e está completa."""
        provider = self.provider.lower()
        
        if provider == 'gemini':
            if not self.gemini:
                raise ValueError("Provedor é 'gemini', mas a seção 'gemini:' está faltando no settings.yml.")
            # A validação da api_key já é feita pelo Pydantic ao ler a env var
        
        elif provider == 'openai':
            if not self.openai:
                raise ValueError("Provedor é 'openai', mas a seção 'openai:' está faltando no settings.yml.")

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
    """Configurações para os serviços de visão computacional."""
    # --- Configurações Gerais de Visão ---
    yolo_model_path: Path
    confidence_threshold: float = Field(0.45, ge=0.1, le=1.0)
    midas_model_path: Path

    # --- CORREÇÃO: Configurações para Reconhecimento Facial (InsightFace) ---
    # As chaves antigas 'deepface_*' foram substituídas para corresponder ao settings.yml.
    face_model: str
    face_model_path: Path
    db_path: Path
    recognition_threshold: float

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
    # MELHORIA: Usa a função robusta para encontrar a raiz do projeto.
    if not config_path.is_absolute():
        try:
            project_root = find_project_root()
            config_path = project_root / config_path
        except FileNotFoundError as e:
            logger.critical(e)
            raise

    if not config_path.exists():
        logger.error(f"Arquivo de configuração não encontrado em: {config_path}")
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    logger.info(f"Carregando configurações de: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Valida a estrutura dos dados carregados usando o modelo Pydantic
        settings = AppSettings(**config_data)
        
        logger.info("Configurações carregadas e validadas com sucesso.")
        return settings

    except yaml.YAMLError as e:
        logger.critical(f"Erro de sintaxe no arquivo de configuração YAML: {e}")
        raise
    except ValidationError as e:
        logger.critical(f"Erro de validação nas configurações. Verifique se 'settings.yml' corresponde à estrutura em 'settings.py'. Erro: {e}")
        raise
    except Exception as e:
        logger.critical(f"Erro inesperado ao carregar as configurações: {e}")
        raise

# --- Instância Global ---
# Tenta carregar as configurações na inicialização do módulo.
# A aplicação não pode continuar se as configurações falharem.
try:
    settings = load_settings()
except (FileNotFoundError, ValueError, ValidationError) as e:
    logger.critical(f"FATAL: Não foi possível carregar as configurações. A aplicação não pode continuar. Erro: {e}")
    settings = None # Deixa explícito que as configurações falharam