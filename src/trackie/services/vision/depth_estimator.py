# src/trackie/services/vision/depth_estimator.py

from pathlib import Path
from typing import Optional
import torch
import cv2
import numpy as np

# Importa o modelo de configurações e o logger do seu projeto
from ...config.settings import AppSettings
from ...utils.logger import get_logger

class DepthEstimator:
    """
    Encapsula o modelo de estimativa de profundidade MiDaS para inferência.

    Esta classe é responsável por carregar o modelo MiDaS a partir de um arquivo local,
    processar os frames de entrada e retornar um mapa de profundidade.
    Ela lida com a inicialização do modelo de forma segura e desabilita o serviço
    automaticamente em caso de falha.
    """
    def __init__(self, settings: AppSettings):
        """
        Inicializa o serviço de estimativa de profundidade.

        Args:
            settings: O objeto de configurações validado (AppSettings) da aplicação.
        """
        self.settings = settings
        self.logger = get_logger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self.transform: Optional[callable] = None
        self.initialized = False

        try:
            self.logger.info(f"Inicializando o serviço de Estimativa de Profundidade no dispositivo: {self.device}")
            self._load_model()
            self.initialized = True
            self.logger.info("Serviço de Estimativa de Profundidade inicializado com sucesso.")

        except Exception as e:
            self.logger.error(f"Erro crítico ao inicializar o DepthEstimator: {e}")
            self.logger.warning("A estimativa de profundidade será desabilitada.")
            self.initialized = False

    def _load_model(self):
        """
        Carrega a arquitetura do modelo MiDaS e os pesos de um arquivo local.
        """
        # Constrói o caminho absoluto para o modelo a partir da raiz do projeto
        project_root = Path(__file__).resolve().parents[3]
        model_path = project_root / self.settings.vision.midas_model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Arquivo do modelo MiDaS não encontrado em '{model_path}'. Execute 'scripts/download_models.py'.")

        # O tipo de modelo é necessário para carregar a arquitetura correta do torch.hub.
        # Assumimos que o nome do arquivo corresponde ao tipo de modelo.
        # Ex: "dpt_levit_224.pt" -> "DPT_LeViT_224"
        model_type = "DPT_LeViT_224"
        self.logger.info(f"Carregando arquitetura do modelo '{model_type}'...")
        
        # 1. Carrega a ARQUITETURA do modelo, mas sem baixar os pesos pré-treinados
        model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, trust_repo=True)

        # 2. Carrega os PESOS do seu arquivo .pt local
        self.logger.info(f"Carregando pesos do arquivo local: {model_path}")
        weights = torch.load(model_path, map_location=self.device)
        model.load_state_dict(weights)
        
        self.model = model
        self.model.to(self.device)
        self.model.eval()

        # 3. Carrega as funções de transformação de imagem correspondentes
        self.logger.info("Carregando transformações de imagem para o MiDaS...")
        transforms_hub = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        # Seleciona a transformação correta com base no tipo de modelo
        self.transform = transforms_hub.dpt_transform if "dpt" in model_type.lower() else transforms_hub.small_transform

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estima o mapa de profundidade de um único frame de imagem.

        Args:
            frame: O frame da imagem em formato NumPy array (BGR).

        Returns:
            Um mapa de profundidade normalizado como um NumPy array, ou None se o serviço
            não estiver inicializado ou em caso de falha.
        """
        if not self.initialized or self.model is None or self.transform is None:
            return None

        try:
            # Converte a cor do frame e aplica as transformações necessárias
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Redimensiona a predição para o tamanho da imagem original
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Retorna o mapa de profundidade como um array numpy
            return prediction.cpu().numpy()

        except Exception as e:
            self.logger.error(f"Erro durante a inferência do MiDaS: {e}", exc_info=True)
            return None

    @staticmethod
    def colorize_depth_map(depth_map: np.ndarray) -> np.ndarray:
        """
        Converte um mapa de profundidade de um canal para uma imagem colorida para visualização.

        Args:
            depth_map: O mapa de profundidade de um canal (valores de ponto flutuante).

        Returns:
            Uma imagem colorida (BGR) representando o mapa de profundidade.
        """
        # Normaliza o mapa de profundidade para o intervalo 0-255
        depth_normalized = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        # Aplica um mapa de cores (ex: INFERNO) para visualização
        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        return colored_depth