# src/trackie/services/llm/factory.py

from ...config.settings import LLMSettings, GeminiSettings, OpenAISettings
from .base import LLMService
from .gemini_service import GeminiService
from .openai_service import OpenAIService
from ...utils.logger import get_logger

logger = get_logger(__name__)

def get_llm_service(llm_settings: LLMSettings) -> LLMService:
    """
    Factory para criar uma instância do serviço de LLM com base nas configurações.

    Args:
        llm_settings: O objeto de configurações Pydantic para o LLM.

    Returns:
        Uma instância de uma classe que implementa a interface LLMService.
    """
    # Acessa o provider diretamente como um atributo do objeto
    provider = llm_settings.provider.lower()
    logger.info(f"Criando serviço de LLM para o provedor: '{provider}'")

    if provider == "gemini":
        # Passa o objeto de configurações específico do Gemini
        if not llm_settings.gemini:
             raise ValueError("Configuração do provedor Gemini ausente no settings.yml")
        return GeminiService(llm_settings.gemini)

    elif provider == "openai":
        # Passa o objeto de configurações específico da OpenAI
        if not llm_settings.openai:
            raise ValueError("Configuração do provedor OpenAI ausente no settings.yml")
        return OpenAIService(llm_settings.openai)
        
    else:
        raise ValueError(f"Provedor de LLM não suportado ou não especificado: '{provider}'")