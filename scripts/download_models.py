# scripts/download_models.py

import os

def clear_console():
    """Limpa o console, compatível com Windows, macOS e Linux."""
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

clear_console()

import sys
from pathlib import Path
import yaml
import torch

# --- Verificação de dependências ---
# A importação do 'FaceDetector' foi removida para evitar o erro.
try:
    import requests
    from tqdm import tqdm
    from deepface import DeepFace
except ImportError:
    print("Por favor, instale as dependências para este script: pip install requests tqdm pyyaml torch deepface")
    sys.exit(1)

# --- Configuração de Path e Logger ---
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from trackie.utils.logger import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)





# --- Mapeamento de URLs para Modelos Específicos ---
MODEL_URLS = {
    # Modelos YOLO
    "yolov5n.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
    "yolov5nu.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5nu.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    
    # Modelos MiDaS
    "dpt_levit_224.pt": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt"
}

def download_file(url: str, destination: Path):
    """Baixa um arquivo com uma barra de progresso, tratando erros."""
    logger.info(f"Baixando de {url} para {destination}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name, total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        logger.info(f"Download de '{destination.name}' concluído.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Falha no download de {url}: {e}")
        if destination.exists():
            os.remove(destination)

def download_model_from_config(config_key: str, model_path_str: str):
    """
    Função genérica para baixar um modelo com base no seu caminho no arquivo de configuração.
    """
    model_path = project_root / model_path_str
    
    if model_path.exists():
        logger.info(f"Modelo '{model_path.name}' já existe. Pulando.")
        return

    model_name = model_path.name
    if model_name not in MODEL_URLS:
        logger.error(f"Não há uma URL de download definida para o modelo '{model_name}' no script.")
        return

    url = MODEL_URLS[model_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, model_path)

def download_deepface_models(config: dict):
    """
    Força o download e cache do modelo de RECONHECIMENTO facial do DeepFace.
    """
    try:
        vision_config = config.get('vision', {})
        model_name = vision_config.get('deepface_model_name', 'VGG-Face')

        logger.info(f"Pré-aquecendo modelo de reconhecimento do DeepFace: '{model_name}'...")
        # Esta função baixa o modelo de reconhecimento principal se ele não existir no cache.
        DeepFace.build_model(model_name)
        logger.info(f"Modelo de reconhecimento '{model_name}' do DeepFace está pronto.")
        
    except Exception as e:
        logger.error(f"Falha ao baixar o modelo de reconhecimento do DeepFace: {e}")

def main():
    """Função principal para executar o download de todos os modelos."""
    clear_console()
    
    logger.info("--- Iniciando o script de download de modelos ---")
    
    config_path = project_root / "config/settings.yml"
    if not config_path.exists():
        logger.critical(f"Arquivo de configuração não encontrado em {config_path}. Não é possível continuar.")
        return
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config or 'vision' not in config:
        logger.critical("O arquivo 'settings.yml' parece estar vazio ou não contém a seção 'vision'.")
        return

    vision_config = config['vision']
    
    # Baixa modelos de visão (YOLO, MiDaS)
    if 'yolo_model_path' in vision_config:
        download_model_from_config('yolo_model_path', vision_config['yolo_model_path'])
    
    if 'midas_model_path' in vision_config:
        download_model_from_config('midas_model_path', vision_config['midas_model_path'])

    # Baixa modelos do DeepFace
    download_deepface_models(config)
    
    logger.info("--- Script de download de modelos concluído ---")


if __name__ == "__main__":
    main()