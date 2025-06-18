import os
import sys
from pathlib import Path
import yaml
import requests
from tqdm import tqdm

def clear_console():
    """Limpa o console, compatível com Windows, macOS e Linux."""
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

# --- Configuração de Paths e Módulos ---
# Adiciona o diretório 'src' ao sys.path para encontrar os módulos do projeto
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# --- Setup do Logger ---
# É importante configurar o logger antes de usá-lo
from trackie.utils.logger import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# --- Mapeamento de URLs para Download de Modelos ---
# Contém URLs para todos os modelos que o script pode baixar.
# Modelos do InsightFace são hospedados no Hugging Face (mirror confiável).
MODEL_URLS = {
    # --- Modelos de Detecção de Objetos (YOLO) ---
    "yolov5n.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
    "yolov5nu.pt": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5nu.pt",
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    
    # --- Modelos de Estimativa de Profundidade (MiDaS) ---
    "dpt_levit_224.pt": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt",
    
    # --- Modelos de Reconhecimento Facial (InsightFace) ---
    # O InsightFace precisa de um modelo de DETECÇÃO e um de RECONHECIMENTO.
    "det_10g.onnx": "https://huggingface.co/Kuvshin/kuvshin8/resolve/main/insightface/models/buffalo_l/det_10g.onnx",
    "w600k_mbf.onnx": "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx",
    "w600k_r50.onnx": "https://huggingface.co/Kuvshin/kuvshin8/resolve/main/insightface/models/buffalo_l/w600k_r50.onnx"

}

# --- Mapeamento de Nomes Amigáveis para Arquivos de Modelo (InsightFace) ---
# Associa o nome do modelo no `settings.yml` aos arquivos .onnx necessários.
INSIGHTFACE_MODELS = {
    "MobileFaceNet": ["det_10g.onnx", "w600k_mbf.onnx"]
    # Adicione outros modelos aqui, se necessário. Ex:
    # "ResNet50": ["det_10g.onnx", "w600k_r50.onnx"]
}

def download_file(url: str, destination: Path):
    """
    Baixa um arquivo de uma URL para um destino local com uma barra de progresso.
    """
    logger.info(f"Baixando de {url} para {destination}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Lança um erro para status HTTP ruins (4xx ou 5xx)
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
        # Remove o arquivo parcialmente baixado em caso de erro
        if destination.exists():
            os.remove(destination)

def download_generic_model(model_path_str: str):
    """
    Função genérica para baixar um modelo com base no seu caminho, se a URL for conhecida.
    """
    model_path = project_root / model_path_str
    
    if model_path.exists():
        logger.info(f"Modelo '{model_path.name}' já existe em {model_path}. Pulando.")
        return

    model_name = model_path.name
    if model_name not in MODEL_URLS:
        logger.error(f"Não há URL de download definida em MODEL_URLS para o modelo '{model_name}'.")
        return

    url = MODEL_URLS[model_name]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, model_path)

def download_insightface_models(config: dict):
    """
    Baixa os modelos de detecção e reconhecimento facial do InsightFace com base
    nas configurações do `settings.yml`.
    """
    logger.info("Verificando modelos do InsightFace...")
    vision_config = config.get('vision', {})
    face_model_name = vision_config.get('face_model')
    face_model_path_str = vision_config.get('face_model_path')

    if not face_model_name or not face_model_path_str:
        logger.warning("Configurações 'face_model' ou 'face_model_path' não encontradas no settings.yml. Pulando download do InsightFace.")
        return

    if face_model_name not in INSIGHTFACE_MODELS:
        logger.error(f"O modelo facial '{face_model_name}' não é conhecido. Modelos disponíveis: {list(INSIGHTFACE_MODELS.keys())}")
        return

    required_files = INSIGHTFACE_MODELS[face_model_name]
    destination_folder = project_root / face_model_path_str
    destination_folder.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Modelo facial configurado: '{face_model_name}'. Verificando arquivos: {required_files}")

    for filename in required_files:
        destination_path = destination_folder / filename
        if destination_path.exists():
            logger.info(f"Arquivo de modelo '{filename}' já existe. Pulando.")
            continue
        
        if filename in MODEL_URLS:
            url = MODEL_URLS[filename]
            download_file(url, destination_path)
        else:
            logger.error(f"Não foi encontrada uma URL para o arquivo de modelo '{filename}'.")

def main():
    """Função principal para orquestrar o download de todos os modelos necessários."""
    clear_console()
    logger.info("--- Iniciando o script de download de modelos ---")
    
    # Verifica a existência do arquivo de configuração
    config_path = project_root / "config/settings.yml"
    if not config_path.exists():
        logger.critical(f"Arquivo de configuração não encontrado em {config_path}. Não é possível continuar.")
        sys.exit(1)
        
    # Carrega as configurações
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not config or 'vision' not in config:
        logger.critical("O arquivo 'settings.yml' parece estar vazio ou não contém a seção 'vision'.")
        return

    vision_config = config['vision']
    
    # --- Download dos modelos configurados ---
    if 'yolo_model_path' in vision_config:
        download_generic_model(vision_config['yolo_model_path'])
    
    if 'midas_model_path' in vision_config:
        download_generic_model(vision_config['midas_model_path'])

    # Baixa os modelos do InsightFace
    download_insightface_models(config)
    
    logger.info("--- Script de download de modelos concluído ---")


if __name__ == "__main__":
    # Mensagem inicial sobre as dependências
    print("Verifique se as dependências do projeto estão instaladas.")
    print("Para o InsightFace, você vai precisar de: pip install insightface onnxruntime")
    print("-" * 50)
    main()
