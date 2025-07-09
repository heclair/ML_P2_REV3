import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import TensorDataset, DataLoader, random_split
from modelo_rna import CNNUsoSolo  # importa o modelo CNN corretamente

def avaliar_modelo(modelo_path="resultados/modelo_rna.pth", dados_path="resultados/"):
    print("🔍 Carregando dados e modelo para avaliação...")

    X = torch.load(os.path.join(dados_path, "X_rna.pt"))
    y = torch.load(os.path.join(dados_path, "y_rna.pt"))

    # Separar conjunto de teste (20%)
    total = len(X)
    test_size = int(0.2 * total)
    val_size = total - test_size

    dataset = TensorDataset(X, y)
    _, test_set = random_split(dataset, [val_size, test_size])
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # Instanciar e carregar o modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = CNNUsoSolo()
    modelo.load_state_dict(torch.load(modelo_path, map_location=device))
    modelo.to(device)
    modelo.eval()

    y_true = []
    y_pred = []

    print("📊 Avaliando...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = modelo(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Salvar relatório
    os.makedirs("resultados", exist_ok=True)
    with open("resultados/relatorio_cnn.txt", "w") as f:
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nMatriz de Confusão:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    print("✅ Avaliação concluída. Resultados salvos em resultados/relatorio_cnn.txt")

    # Plotar matriz de confusão
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Vegetação", "Não Vegetação", "Água"], cmap="Blues"
    )
    plt.title("Matriz de Confusão - CNN")
    plt.tight_layout()
    plt.savefig("resultados/matriz_confusao_cnn.png")
    plt.show()

# 👇 Executar diretamente via terminal
if __name__ == "__main__":
    avaliar_modelo()
