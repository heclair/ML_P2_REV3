from scripts.preprocessamento import carregar_dados
from scripts.classificadores import treinar_random_forest, treinar_svm

def main():
    # 1. Carrega e prepara os dados
    print("📦 Carregando dados...")
    X, y = carregar_dados("img_sentinel/")

    # 2. Treina Random Forest
    print("\n🏁 Iniciando classificação com Random Forest...")
    treinar_random_forest(X, y)

    # 3. Treina SVM
    print("\n🏁 Iniciando classificação com SVM...")
    treinar_svm(X, y)

if __name__ == "__main__":
    main()
