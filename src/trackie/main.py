# src/trackie/main.py

import asyncio
import argparse

# Importa o essencial: a classe principal da aplicação e as funções de setup.
from .core.application import Application
from .config.settings import settings, load_settings
from .utils.logger import setup_logging, get_logger


async def main():
    """Ponto de entrada que configura e executa a aplicação Trackie."""
    # 1. Configuração Inicial
    # O logger e as configurações já são inicializados quando o módulo é importado.
    setup_logging()
    logger = get_logger(__name__)

    # Verifica se as configurações foram carregadas corretamente.
    if not settings:
        logger.critical("Falha crítica ao carregar as configurações. A aplicação não pode continuar.")
        return

    # 2. Argumentos da Linha de Comando
    parser = argparse.ArgumentParser(description="Trackie - Assistente IA com percepção multimodal.")
    parser.add_argument(
        "--show_preview",
        action="store_true",
        help="Mostra uma janela com o preview da câmera e informações de depuração visual."
    )
    args = parser.parse_args()

    # 3. Criação e Execução da Aplicação
    # A classe Application é a única responsável por instanciar e gerenciar seus serviços.
    # Passamos apenas as configurações e o argumento do preview.
    app_instance = None
    try:
        logger.info("Criando a instância principal da aplicação...")
        app_instance = Application(
            settings=settings,
            show_preview=args.show_preview
        )

        logger.info("Iniciando Trackie...")
        await app_instance.run()

    except KeyboardInterrupt:
        logger.info("\nInterrupção pelo teclado recebida. Encerrando de forma organizada...")
    except Exception:
        # Pega qualquer outra exceção não tratada para um log detalhado.
        logger.critical("Erro fatal e inesperado na execução da aplicação.", exc_info=True)
    finally:
        # Garante que, independentemente do que aconteça, a aplicação tente parar graciosamente.
        if app_instance and not app_instance.is_stopped():
              logger.info("Sinalizando parada para todas as tarefas...")
              await app_instance.stop()
        logger.info("Aplicação finalizada.")


if __name__ == "__main__":
    # Inicia o loop de eventos assíncrono para a função main
    asyncio.run(main())