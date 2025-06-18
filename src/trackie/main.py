# src/trackie/main.py

import asyncio
import argparse
import traceback
from pathlib import Path

# Importa o objeto de configurações e os modelos Pydantic
from .config.settings import load_settings
from .utils.logger import setup_logging, get_logger

# Importa as classes e fábricas de todos os nossos módulos
from .core.application import Application
from .services.llm.factory import get_llm_service
from .services.vision.object_detector import ObjectDetector
from .services.vision.depth_estimator import DepthEstimator
from .services.vision.face_recognizer import FaceRecognizer
from .services.audio.microphone import MicrophoneService
from .services.audio.speaker import SpeakerService
from .services.video.camera import CameraService
from .ui.preview import PreviewWindow
from .tools.registry import ToolRegistry


async def main():
    """Ponto de entrada principal que monta e executa a aplicação Trackie."""
    # 1. Configuração Inicial
    setup_logging()
    logger = get_logger(__name__)

    try:
        settings = load_settings()
        if not settings:
            # A função load_settings já loga o erro, então aqui apenas saímos.
            return
    except Exception as e:
        logger.critical(f"Falha crítica ao carregar as configurações. A aplicação não pode continuar. Erro: {e}")
        return

    # 2. Argumentos da Linha de Comando
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA visual e auditivo.")
    parser.add_argument(
        "--show_preview", action="store_true",
        help="Mostra janela com preview da câmera e detecções."
    )
    args = parser.parse_args()

    # --- 3. Montagem dos Serviços (Injeção de Dependência) ---
    app_instance = None
    try:
        logger.info("Iniciando a montagem dos serviços da aplicação...")
        
        # O objeto 'settings' é passado diretamente, garantindo segurança de tipos.
        
        # Serviços de IA e Hardware
        llm_service = get_llm_service(settings.llm)
        object_detector = ObjectDetector(settings)
        depth_estimator = DepthEstimator(settings)
        face_recognizer = FaceRecognizer(settings)
        microphone_service = MicrophoneService(settings)
        speaker_service = SpeakerService(settings)
        
        # Serviços de UI (Opcional)
        preview_service = PreviewWindow() if args.show_preview else None
        
        # O CameraService é criado sem o 'shared_state' inicialmente.
        camera_service = CameraService(settings, object_detector, preview_service)

        # Montagem do Registry de Ferramentas
        tool_registry = ToolRegistry(
            settings=settings,
            face_recognizer=face_recognizer,
            object_detector=object_detector,
            depth_estimator=depth_estimator
        )

        # 4. Criação da Instância Principal da Aplicação
        logger.info("Todos os serviços montados. Criando a instância principal da aplicação.")
        app_instance = Application(
            settings=settings,
            llm_service=llm_service,
            tool_registry=tool_registry,
            camera_service=camera_service,
            mic_service=microphone_service,
            speaker_service=speaker_service,
            preview_service=preview_service,
            depth_estimator=depth_estimator,
            face_recognizer=face_recognizer,
            object_detector=object_detector
        )

        # 5. Injeção do Estado Compartilhado (resolvendo a dependência circular)
        # Após a 'app_instance' ser criada, nós a injetamos nos serviços que precisam dela.
        # Isso requer que os serviços tenham um método como 'set_shared_state'.
        logger.info("Injetando estado compartilhado nos serviços...")
        tool_registry.set_shared_state(app_instance)
        camera_service.set_shared_state(app_instance)
        
        # 6. Execução da Aplicação
        logger.info("Iniciando Trackie...")
        await app_instance.run()

    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado recebida. Encerrando...")
    except Exception:
        logger.critical("Erro fatal e inesperado no nível principal da aplicação.", exc_info=True)
    finally:
        if app_instance and not app_instance.is_stopped():
             logger.info("Sinalizando parada para as tarefas...")
             await app_instance.stop()
        logger.info("Aplicação finalizada.")


if __name__ == "__main__":
    # Inicia o loop de eventos assíncrono para a função main
    asyncio.run(main())