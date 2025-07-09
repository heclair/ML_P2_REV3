import os
from scripts.preprocessamento_rna import gerar_patches
from scripts.modelo_rna import treinar_rna

def main():
    salvar_em = "./resultados/"
    arquivo_X = os.path.join(salvar_em, "X_rna.pt")
    arquivo_y = os.path.join(salvar_em, "y_rna.pt")

    print("🚧 Etapa 1: Geração de patches para RNA")

    if not os.path.exists(arquivo_X) or not os.path.exists(arquivo_y):
        gerar_patches(max_amostras=100_000)
    else:
        print("✔️ Patches já existem. Pulando geração...")

    print("\n🚀 Etapa 2: Treinamento do modelo CNN")
    try:
        treinar_rna(batch_size=64, epochs=10, lr=0.001)
        print("\n✅ Treinamento finalizado.")
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")

if __name__ == "__main__":
    main()
