import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

# --- Imports dos Módulos do Projeto ---
from ..config.settings import load_settings
from ..services.llm.factory import LLMFactory
from ..services.video.camera import Camera
from ..services.audio.microphone import Microphone
from ..services.audio.speaker import Speaker
from ..services.vision.object_detector import ObjectDetector
from ..services.vision.depth_estimator import DepthEstimator
from ..services.vision.face_recognizer import FaceRecognizer
from ..tools.registry import ToolRegistry
from ..ui.preview import Preview
from ..utils.logger import get_logger

logger = get_logger(__name__)

class Application:
    """
    Orquestra o fluxo principal da aplicação Trackie, gerenciando a inicialização
    de todos os serviços e a interação assíncrona entre eles.
    """
    def __init__(self):
        """
        Inicializa a aplicação, carrega as configurações e cria todos os serviços.
        """
        logger.info("Inicializando a aplicação Trackie...")
        self.settings = load_settings()
        self.trckuser = self.settings.get("user", {}).get("name", "usuário")
        
        # Filas para comunicação assíncrona
        self.media_to_llm_queue = asyncio.Queue(maxsize=150)
        self.audio_from_llm_queue = asyncio.Queue()

        # Eventos de sincronização
        self.stop_event = asyncio.Event()
        
        # Inicializa todos os serviços
        self._initialize_services()

    def _initialize_services(self):
        """
        Cria e configura todas as instâncias de serviços com base no arquivo de settings.
        """
        logger.info("Inicializando todos os serviços...")
        
        # --- Configurações ---
        llm_config = self.settings.get('llm', {})
        vision_config = self.settings.get('vision', {})
        video_config = self.settings.get('video', {})
        audio_config = self.settings.get('audio', {})
        paths_config = self.settings.get('paths', {})

        # --- Serviços de Visão (agora inicializados aqui) ---
        logger.info("Carregando modelos de visão...")
        self.object_detector = ObjectDetector(
            vision_config['yolo_model_path'], 
            vision_config['confidence_threshold']
        )
        self.depth_estimator = DepthEstimator(vision_config['midas_model_path'])
        
        # *** AQUI ESTÁ A MUDANÇA PRINCIPAL: Usando o novo FaceRecognizer ***
        self.face_recognizer = FaceRecognizer(
            model_path=vision_config['face_model_path'],
            db_path=vision_config['db_path'],
            recognition_threshold=vision_config['recognition_threshold']
        )
        logger.info("Serviço de reconhecimento facial (InsightFace) pronto.")

        # --- Serviços de Mídia e IA ---
        self.llm_service = LLMFactory.create_service(config=llm_config)
        
        self.camera_service = Camera(
            video_config=video_config,
            object_detector=self.object_detector,
            depth_estimator=self.depth_estimator,
            face_recognizer=self.face_recognizer
        ) if video_config.get("provider") else None
        
        self.microphone_service = Microphone(config=audio_config)
        self.speaker_service = Speaker(config=audio_config)
        
        # --- Ferramentas e UI ---
        # O ToolRegistry agora recebe as instâncias de serviço diretamente
        self.tool_registry = ToolRegistry(
            llm_service=self.llm_service,
            camera_service=self.camera_service,
            # Passamos o face_recognizer diretamente para as ferramentas
            face_recognizer_service=self.face_recognizer, 
            config_paths=paths_config,
        )
        # O thinking_event é gerenciado pelo ToolRegistry
        self.thinking_event = self.tool_registry.thinking_event

        self.preview_window = Preview(self.tool_registry) if self.camera_service else None
        
        logger.info("Todos os serviços foram inicializados com sucesso.")

    async def _process_llm_responses(self):
        """Consome as respostas do LLM (texto, áudio, chamadas de função) e as encaminha."""
        logger.info("Processador de respostas do LLM iniciado.")
        try:
            async for response in self.llm_service.receive():
                if self.stop_event.is_set():
                    break

                if response.type == "audio" and response.data:
                    await self.audio_from_llm_queue.put(response.data)
                
                elif response.type == "text" and response.data:
                    print(response.data, end="", flush=True)

                elif response.type == "function_call":
                    logger.info(f"Recebida solicitação para ferramenta: {response.data.name}")
                    asyncio.create_task(
                        self.tool_registry.execute(response.data.name, response.data.args)
                    )
        except asyncio.CancelledError:
            logger.info("Processador de respostas do LLM cancelado.")
        except Exception:
            logger.exception("Erro crítico no processador de respostas do LLM.")
            self.stop_event.set()

    async def run(self):
        """O loop principal que inicia e supervisiona todas as tarefas da aplicação."""
        logger.info("Iniciando o loop principal da aplicação (Application.run)...")
        try:
            prompt_path = Path(self.settings["llm"]["system_prompt_path"])
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read().replace("{TRCKUSER}", self.trckuser)

            await self.llm_service.connect(
                system_prompt=system_prompt,
                tools=self.tool_registry.get_definitions()
            )

            async with asyncio.TaskGroup() as tg:
                logger.info("Iniciando o grupo de tarefas principal da aplicação...")
                
                tg.create_task(self._process_llm_responses())
                tg.create_task(self.llm_service.send_media_stream(self.media_to_llm_queue, self.thinking_event))
                tg.create_task(self.microphone_service.stream(self.media_to_llm_queue, self.stop_event, self.thinking_event))
                tg.create_task(self.speaker_service.stream(self.audio_from_llm_queue, self.stop_event))

                if self.camera_service:
                    tg.create_task(self.camera_service.stream(self.media_to_llm_queue, self.stop_event))
                
                if self.preview_window:
                    tg.create_task(self.preview_window.run(self.stop_event))

                logger.info("Todas as tarefas foram iniciadas. Trackie está operacional.")

        except Exception:
            logger.exception("Erro fatal no loop de execução da aplicação.")
        finally:
            logger.info("Iniciando o processo de limpeza da aplicação...")
            self.stop_event.set()
            await self._cleanup_resources()
            logger.info("Aplicação finalizada.")

    async def _cleanup_resources(self):
        """Realiza a limpeza final de todos os recursos e serviços."""
        logger.info("Limpando recursos...")
        if self.audio_from_llm_queue:
            await self.audio_from_llm_queue.put(None)
        if self.llm_service:
            await self.llm_service.close()
        if self.preview_window:
            self.preview_window.destroy()
        logger.info("Limpeza de recursos concluída.")
