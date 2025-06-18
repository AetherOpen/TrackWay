# scripts/setup_config.py

import os
from pathlib import Path
import yaml
import sys

def ask_question(prompt: str, default: str = None) -> str:
    """Faz uma pergunta ao usuário e retorna a resposta."""
    if default:
        prompt_with_default = f"{prompt} (padrão: {default}): "
    else:
        prompt_with_default = f"{prompt}: "
    
    response = input(prompt_with_default).strip()
    return response or default

def create_settings_yml(config_path: Path):
    """Guia o usuário na criação do arquivo settings.yml."""
    print("\n--- Configuração do Trackie ---")
    print("Vamos criar seu arquivo 'config/settings.yml'.")
    
    user_name = ask_question("Qual é o seu nome ou apelido?", "Usuário")
    
    settings_data = {
        'user': {'name': user_name},
        'llm': {
            'provider': 'gemini',
            'system_prompt_path': 'config/prompts.yml',
            'gemini': {
                'api_key': '${GEMINI_API_KEY}',
                'model': 'models/gemini-1.5-flash-latest',
                'temperature': 0.2
            },
            'openai': {
                'api_key': '${OPENAI_API_KEY}',
                'model': 'gpt-4o',
                'temperature': 0.7
            }
        },
        'video': {
            'provider': 'opencv',
            'device_index': 0,
            'fps': 2.0,
            'jpeg_quality': 60
        },
        'audio': {
            'chunk_size': 1024,
            'send_sample_rate': 16000,
            'receive_sample_rate': 24000,
            'channels': 1
        },
        'vision': {
            'yolo_model_path': 'data/models/yolov8n.pt',
            'confidence_threshold': 0.45,
            'midas_model_type': 'DPT_SwinV2_L_384',
            'deepface_db_path': 'data/user_data/known_faces',
            'deepface_model_name': 'VGG-Face',
            'deepface_detector_backend': 'opencv',
            'deepface_distance_metric': 'cosine'
        },
        'paths': {
            'data': 'data',
            'tool_definitions': 'config/tool_definitions.json',
            'danger_sound': 'data/assets/sounds/trackie_danger.wav'
        }
    }
    
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(settings_data, f, sort_keys=False, indent=2)
        
    print(f"\nArquivo 'config/settings.yml' criado com sucesso!")

def create_env_file(env_path: Path):
    """Guia o usuário na criação do arquivo .env para as chaves de API."""
    print("\n--- Configuração das Chaves de API ---")
    print("Agora, vamos configurar suas chaves de API em um arquivo .env.")
    print("Este arquivo é privado e não deve ser enviado para o GitHub.")
    
    gemini_key = ask_question("Por favor, insira sua chave da API do Google Gemini")
    openai_key = ask_question("Por favor, insira sua chave da API da OpenAI (opcional, pode deixar em branco)")
    
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write("# Variáveis de ambiente para o projeto Trackie\n")
        if gemini_key:
            f.write(f'GEMINI_API_KEY="{gemini_key}"\n')
        if openai_key:
            f.write(f'OPENAI_API_KEY="{openai_key}"\n')
            
    print(f"\nArquivo '.env' criado com sucesso!")
    print("Lembre-se de adicionar '.env' ao seu arquivo .gitignore se ainda não o fez.")

def main():
    """Função principal para executar o setup interativo."""
    project_root = Path(__file__).resolve().parent.parent
    
    settings_path = project_root / "config/settings.yml"
    env_path = project_root / ".env"
    
    if settings_path.exists() and env_path.exists():
        overwrite = ask_question(
            "Arquivos de configuração ('settings.yml' e '.env') já existem. Deseja sobrescrevê-los? (s/n)", "n"
        ).lower()
        if overwrite != 's':
            print("Operação cancelada.")
            return
            
    create_settings_yml(settings_path)
    create_env_file(env_path)
    
    print("\n--- Configuração Concluída! ---")
    print("Próximos passos:")
    print("1. Execute 'pip install -r requirements.txt' para instalar as dependências.")
    print("2. Execute 'python scripts/download_models.py' para baixar os modelos de IA.")
    print("3. Execute 'python -m src.trackie.main' para iniciar a aplicação.")

if __name__ == "__main__":
    try:
        import yaml
    except ImportError:
        print("Por favor, instale a dependência para este script: pip install pyyaml")
        sys.exit(1)
        
    main()
