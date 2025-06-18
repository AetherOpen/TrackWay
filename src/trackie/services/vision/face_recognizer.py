import logging
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from ...utils.logger import get_logger as logger

class FaceRecognizerService:
    """
    Serviço para reconhecimento facial usando a biblioteca InsightFace.

    Esta classe carrega um modelo de análise facial, indexa um banco de dados de rostos conhecidos
    e pode reconhecer esses rostos em um determinado frame de imagem.
    """

    def __init__(self, db_path: str, recognition_threshold: float):
        """
        Inicializa o serviço de reconhecimento facial.

        Args:
            db_path (str): Caminho para o diretório contendo imagens de rostos conhecidos.
            recognition_threshold (float): Limiar de similaridade para considerar um rosto reconhecido.
        """
        self.db_path = db_path
        self.recognition_threshold = recognition_threshold
        self.known_faces = []

        # Inicializa o modelo FaceAnalysis. Ele baixará os modelos necessários na primeira execução.
        # Usamos 'buffalo_l' que é um pacote completo com detecção, alinhamento e reconhecimento (MobileFaceNet).
        logger.info("Inicializando o modelo InsightFace...")
        try:
            self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Modelo InsightFace carregado com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo InsightFace: {e}")
            raise

        # Indexa os rostos conhecidos que já existem no banco de dados
        self._index_known_faces()

    def _index_known_faces(self):
        """
        Lê todas as imagens do banco de dados de rostos, extrai seus embeddings
        e os armazena em memória para comparação rápida.
        """
        logger.info(f"Indexando rostos conhecidos de '{self.db_path}'...")
        if not os.path.exists(self.db_path):
            logger.warning(f"Diretório do banco de dados de rostos não encontrado: '{self.db_path}'. Criando...")
            os.makedirs(self.db_path)

        for filename in os.listdir(self.db_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                person_name = os.path.splitext(filename)[0]
                filepath = os.path.join(self.db_path, filename)
                
                try:
                    img = cv2.imread(filepath)
                    if img is None:
                        logger.warning(f"Não foi possível ler a imagem: {filepath}")
                        continue

                    faces = self.app.get(img)
                    if faces and len(faces) == 1:
                        embedding = faces[0].normed_embedding
                        self.known_faces.append({"name": person_name, "embedding": embedding})
                        logger.info(f"Rosto de '{person_name}' indexado.")
                    elif not faces:
                        logger.warning(f"Nenhum rosto encontrado na imagem de '{person_name}' em {filepath}.")
                    else:
                        logger.warning(f"Múltiplos rostos encontrados na imagem de '{person_name}' em {filepath}. Usando apenas o primeiro.")
                        embedding = faces[0].normed_embedding
                        self.known_faces.append({"name": person_name, "embedding": embedding})


                except Exception as e:
                    logger.error(f"Erro ao processar o rosto de '{person_name}' em {filepath}: {e}")
        
        logger.info(f"Indexação concluída. {len(self.known_faces)} rostos conhecidos carregados.")

    def recognize_faces(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Detecta e reconhece rostos em um frame.

        Args:
            bgr_frame (np.ndarray): O frame da câmera no formato BGR.

        Returns:
            list[dict]: Uma lista de dicionários, onde cada um contém o nome
                        e a caixa delimitadora (bbox) do rosto reconhecido.
                        Ex: [{'name': 'John', 'box': (x1, y1, x2, y2)}]
        """
        if not self.known_faces:
            return []

        recognized_people = []
        try:
            faces_in_frame = self.app.get(bgr_frame)
        except Exception as e:
            logger.error(f"Erro ao obter rostos do frame: {e}")
            return []

        for face in faces_in_frame:
            current_embedding = face.normed_embedding
            best_match_name = None
            highest_similarity = -1

            # Compara o rosto atual com todos os rostos conhecidos
            for known_face in self.known_faces:
                # Usa produto escalar para similaridade de cosseno, pois os vetores são normalizados
                similarity = np.dot(current_embedding, known_face["embedding"])
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_name = known_face["name"]
            
            # Se a melhor correspondência estiver acima do nosso limiar, consideramos reconhecida
            if highest_similarity > self.recognition_threshold:
                box = face.bbox.astype(int)
                recognized_people.append({"name": best_match_name, "box": tuple(box)})
                logger.debug(f"Rosto reconhecido: {best_match_name} com similaridade {highest_similarity:.2f}")

        return recognized_people

    def add_known_face(self, person_name: str, bgr_frame: np.ndarray) -> bool:
        """
        Salva um novo rosto conhecido no banco de dados e o indexa em tempo real.

        Args:
            person_name (str): O nome da pessoa a ser salva.
            bgr_frame (np.ndarray): Um frame contendo o rosto da pessoa.

        Returns:
            bool: True se o rosto foi adicionado com sucesso, False caso contrário.
        """
        try:
            faces = self.app.get(bgr_frame)
            if not faces:
                logger.error(f"Nenhum rosto detectado na imagem para salvar '{person_name}'.")
                return False
            
            if len(faces) > 1:
                logger.warning(f"Múltiplos rostos detectados. Salvando o maior deles para '{person_name}'.")
                # Lógica para pegar o maior rosto pela área do bbox
                faces.sort(key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]), reverse=True)

            face_to_save = faces[0]
            embedding = face_to_save.normed_embedding

            # Salva a imagem no disco
            filename = f"{person_name}.jpg"
            filepath = os.path.join(self.db_path, filename)
            cv2.imwrite(filepath, bgr_frame)
            
            # Adiciona ao índice em memória
            self.known_faces.append({"name": person_name, "embedding": embedding})
            
            logger.info(f"Novo rosto '{person_name}' salvo em {filepath} e adicionado ao índice.")
            return True
        except Exception as e:
            logger.error(f"Falha ao adicionar novo rosto conhecido '{person_name}': {e}")
            return False