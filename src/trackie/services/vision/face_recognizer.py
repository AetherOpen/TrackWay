import os
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Import relativo para o logger do projeto
from ...utils.logger import get_logger

# Configura o logger para este módulo
logger = get_logger(__name__)

class FaceRecognizer:
    """
    Serviço para reconhecimento facial usando a biblioteca InsightFace.

    Esta classe é responsável por:
    1. Carregar modelos de detecção e reconhecimento facial a partir de um caminho local.
    2. Indexar um banco de dados de rostos conhecidos a partir de imagens.
    3. Reconhecer rostos em um frame de vídeo e compará-los com o banco de dados.
    4. Adicionar novos rostos ao banco de dados.
    """

    def __init__(self, model_path: str, db_path: str, recognition_threshold: float):
        """
        Inicializa o serviço de reconhecimento facial.

        Args:
            model_path (str): Caminho para a pasta que contém os modelos .onnx do InsightFace.
            db_path (str): Caminho para o diretório contendo imagens de rostos conhecidos.
            recognition_threshold (float): Limiar de similaridade de cosseno para um rosto ser considerado uma correspondência.
        """
        self.model_path = Path(model_path)
        self.db_path = Path(db_path)
        self.recognition_threshold = recognition_threshold
        self.known_faces = []  # Cache em memória para os embeddings de rostos conhecidos

        if not self.model_path.exists():
            logger.critical(f"A pasta de modelos do InsightFace não foi encontrada em: {self.model_path}")
            raise FileNotFoundError(f"Pasta de modelos não encontrada: {self.model_path}")

        logger.info("Inicializando o serviço de reconhecimento facial com InsightFace...")
        try:
            # Inicializa o FaceAnalysis, especificando o nome do conjunto de modelos ('antelopev2' para os que baixamos)
            # e o 'root' para o caminho local, forçando o uso dos modelos baixados.
            self.app = FaceAnalysis(
                name='antelopev2',
                root=self.model_path,
                providers=['CPUExecutionProvider'] # Use 'CUDAExecutionProvider' se tiver GPU configurada
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("Modelo InsightFace carregado com sucesso a partir de arquivos locais.")
        except Exception as e:
            logger.error(f"Falha ao carregar o modelo InsightFace de '{self.model_path}': {e}")
            logger.error("Verifique se os arquivos .onnx corretos (ex: det_10g.onnx, w600k_mbf.onnx) estão na pasta.")
            raise

        # Indexa os rostos conhecidos que já existem no banco de dados
        self._index_known_faces()

    def _index_known_faces(self):
        """
        Lê todas as imagens do banco de dados, extrai seus embeddings
        e os armazena em memória para comparação rápida.
        """
        logger.info(f"Indexando rostos conhecidos de '{self.db_path}'...")
        if not self.db_path.exists():
            logger.warning(f"Diretório do banco de dados de rostos não encontrado. Criando em: '{self.db_path}'")
            self.db_path.mkdir(parents=True, exist_ok=True)

        for filepath in self.db_path.glob('*.*'):
            if filepath.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            person_name = filepath.stem  # Nome do arquivo sem a extensão
            try:
                img = cv2.imread(str(filepath))
                if img is None:
                    logger.warning(f"Não foi possível ler a imagem: {filepath}")
                    continue

                faces = self.app.get(img)
                if faces:
                    # Usa o embedding do primeiro (e idealmente único) rosto na imagem
                    embedding = faces[0].normed_embedding
                    self.known_faces.append({"name": person_name, "embedding": embedding})
                    logger.debug(f"Rosto de '{person_name}' indexado.")
                else:
                    logger.warning(f"Nenhum rosto encontrado na imagem de referência para '{person_name}' em {filepath}.")

            except Exception as e:
                logger.error(f"Erro ao processar o arquivo de rosto '{filepath}': {e}")
        
        logger.info(f"Indexação concluída. {len(self.known_faces)} rostos conhecidos carregados.")

    def recognize_faces(self, bgr_frame: np.ndarray) -> list[dict]:
        """
        Detecta e reconhece rostos em um frame de vídeo.

        Args:
            bgr_frame (np.ndarray): O frame da câmera no formato BGR.

        Returns:
            list[dict]: Uma lista de dicionários, cada um contendo o nome e a
                        caixa delimitadora (bbox) de um rosto reconhecido.
                        Ex: [{'name': 'Aether', 'box': (x1, y1, x2, y2)}]
        """
        if not self.known_faces:
            return []  # Retorna cedo se não houver rostos para comparar

        recognized_people = []
        try:
            faces_in_frame = self.app.get(bgr_frame)
        except Exception as e:
            logger.error(f"Erro ao analisar o frame de vídeo com InsightFace: {e}")
            return []

        for face in faces_in_frame:
            current_embedding = face.normed_embedding
            best_match_name = "desconhecido"
            highest_similarity = 0.0

            # Compara o rosto detectado com todos os rostos conhecidos
            for known_face in self.known_faces:
                # Calcula a similaridade de cosseno (produto escalar de vetores normalizados)
                similarity = np.dot(current_embedding, known_face["embedding"])
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    if similarity > self.recognition_threshold:
                        best_match_name = known_face["name"]
            
            # Adiciona a pessoa à lista se foi reconhecida
            if best_match_name != "desconhecido":
                box = face.bbox.astype(int)
                recognized_people.append({"name": best_match_name, "box": tuple(box)})
                logger.debug(f"Rosto reconhecido: {best_match_name} com similaridade {highest_similarity:.2f}")

        return recognized_people

    def add_known_face(self, person_name: str, bgr_frame: np.ndarray) -> bool:
        """
        Extrai um rosto de um frame, salva-o como uma imagem cortada no banco de dados
        e o adiciona ao índice de reconhecimento em tempo de execução.

        Args:
            person_name (str): O nome da pessoa a ser salva.
            bgr_frame (np.ndarray): O frame contendo o rosto da pessoa.

        Returns:
            bool: True se o rosto foi adicionado com sucesso, False caso contrário.
        """
        try:
            faces = self.app.get(bgr_frame)
            if not faces:
                logger.error(f"Nenhum rosto detectado na imagem para salvar '{person_name}'.")
                return False
            
            # Se múltiplos rostos forem detectados, usa o maior (com maior área de bbox)
            if len(faces) > 1:
                logger.warning(f"Múltiplos rostos detectados. Usando o maior deles para '{person_name}'.")
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

            face_to_save = faces[0]
            embedding = face_to_save.normed_embedding

            # Define o caminho do arquivo e verifica se já existe
            filepath = self.db_path / f"{person_name}.jpg"
            if filepath.exists():
                logger.warning(f"Já existe um rosto para '{person_name}'. A imagem será sobrescrita.")

            # Corta o rosto do frame original para salvar uma imagem limpa
            x1, y1, x2, y2 = face_to_save.bbox.astype(int)
            padding = 20 # Adiciona um pouco de espaço ao redor do rosto
            face_img = bgr_frame[max(0, y1-padding):y2+padding, max(0, x1-padding):x2+padding]

            # Salva a imagem cortada no disco
            cv2.imwrite(str(filepath), face_img)
            
            # Atualiza o índice em memória. Remove o antigo se estiver sobrescrevendo.
            self.known_faces = [face for face in self.known_faces if face["name"] != person_name]
            self.known_faces.append({"name": person_name, "embedding": embedding})
            
            logger.info(f"Novo rosto '{person_name}' salvo em {filepath} e adicionado ao índice.")
            return True

        except Exception as e:
            logger.error(f"Falha ao adicionar novo rosto conhecido para '{person_name}': {e}")
            return False
