import fire
from src.indexing.builder import IndexBuilder
class RAGCli:
    """CLI principal para el proyecto RAG."""

    def index(self, repo_path: str, save_path: str = "data/processed/bm25_index") -> None:
        """Indexa el repositorio y guarda el modelo BM25."""
        # TODO: Instanciar el builder y ejecutarlo
        pass

def main():
    # fire.Fire convierte los métodos de la clase RAGCli en comandos de terminal
    fire.Fire(RAGCli)

if __name__ == "__main__":
    main()