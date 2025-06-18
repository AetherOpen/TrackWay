# src/trackie/tools/handlers.py

import os
from typing import Any, Dict, Optional

# Importa as funções de cálculo que acabamos de criar
from ..processing import calculations
from ..utils.logger import get_logger
import asyncio

logger = get_logger(__name__)

class ToolHandlers:
    """
    Contém a implementação lógica para cada ferramenta que pode ser chamada pelo LLM.
    """
    def __init__(
        self,
        # TODO: Substituir 'Any' por classes de interface
        face_recognizer: Any,
        object_detector: Any,
        depth_estimator: Any,
        shared_state: Any, # A instância da Application
        settings: Dict[str, Any]
    ):
        """
        Inicializa os handlers com os serviços necessários.
        """
        self.face_recognizer = face_recognizer
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        self.shared_state = shared_state
        self.settings = settings
        self.user_name = settings.get("user", {}).get("name", "usuário")

    async def handle_save_known_face(self, person_name: str) -> Dict[str, Any]:
        """Lida com a lógica de salvar um rosto conhecido."""
        logger.info(f"Executando 'save_known_face' para: {person_name}")
        
        async with self.shared_state.frame_lock:
            frame = self.shared_state.latest_bgr_frame
        
        if frame is None:
            return {"result": "Desculpe, não consigo ver nada no momento para salvar o rosto."}

        # Delega a lógica de salvamento para o serviço de reconhecimento facial
        success = await asyncio.to_thread(self.face_recognizer.save_face, frame, person_name)
        
        if success:
            result_message = f"O rosto de {person_name} foi salvo com sucesso."
        else:
            result_message = f"Não consegui detectar um rosto claro para salvar para {person_name}."
            
        return {"result": result_message}

    async def handle_identify_person_in_front(self) -> Dict[str, Any]:
        """
        Identifica uma pessoa no campo de visão da câmera.
        Retorna o nome da pessoa se reconhecida, ou uma mensagem de falha.
        """
        if not self.shared_state.latest_bgr_frame is not None:
            return "Não consegui obter uma imagem da câmera para análise."

        frame = self.shared_state.latest_bgr_frame
        
        # O método agora retorna uma lista de dicts
        recognized_faces = self.face_recognizer.recognize_faces(frame)

        if not recognized_faces:
            return "Não reconheci ninguém que eu conheça no momento."
        
        # Pega o nome do primeiro rosto reconhecido na lista
        # O novo formato é muito mais simples de acessar
        name = recognized_faces[0]["name"].replace("_", " ").title()
        
        if len(recognized_faces) > 1:
            # Se houver mais de uma pessoa, podemos mencioná-las
            # (ou apenas focar na primeira, como estamos fazendo agora)
            return f"Eu vejo {name} e outras pessoas."

        return f"A pessoa na sua frente é {name}."

    async def handle_locate_object(self, object_name: str) -> Dict[str, Any]:
        """Lida com a lógica de localizar um objeto e estimar sua distância."""
        logger.info(f"Executando 'locate_object' para: {object_name}")

        with self.shared_state.frame_lock:
            frame = self.shared_state.latest_bgr_frame
            yolo_results = self.shared_state.latest_yolo_results
        
        if frame is None or yolo_results is None:
            return {"result": "Não estou conseguindo processar a imagem no momento."}

        # 1. Usa o módulo de cálculo para encontrar o objeto
        yolo_names = self.object_detector.model.names
        best_match = calculations.find_best_yolo_match(object_name, yolo_results, yolo_names)

        if not best_match:
            return {"result": f"Não consegui encontrar um(a) {object_name} na imagem."}
        
        bbox, _, detected_class = best_match
        
        # 2. Usa o módulo de cálculo para obter informações adicionais
        direction = calculations.estimate_direction_from_bbox(bbox, frame.shape[1])
        on_surface = calculations.check_if_on_surface(bbox, yolo_results, yolo_names)
        
        # 3. Estima a distância
        steps_str = ""
        depth_map = await asyncio.to_thread(self.depth_estimator.estimate, frame)
        if depth_map is not None:
            steps = calculations.estimate_distance_in_steps(depth_map, bbox)
            if steps:
                steps_str = f"a aproximadamente {steps} passo{'s' if steps > 1 else ''}"

        # 4. Monta a resposta final
        response_parts = [f"O {detected_class}"]
        if on_surface:
            response_parts.append("está sobre uma superfície")
        if steps_str:
            response_parts.append(steps_str)
        response_parts.append(direction)
        
        # Constrói uma frase mais natural
        if len(response_parts) > 2:
            result_message = f"{response_parts[0]} {response_parts[1]}, {', '.join(response_parts[2:])}."
        else:
            result_message = f"{response_parts[0]} está {direction}."

        return {"result": result_message}