import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class CNNUsoSolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),  # Entrada: 10 bandas
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # 3 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x

def treinar_rna(batch_size=64, epochs=30, lr=0.001, early_stop_patience=3):
    print("📥 Carregando dados...")
    X = torch.load("resultados/X_rna.pt")
    y = torch.load("resultados/y_rna.pt")

    dataset = TensorDataset(X, y)

    # Split (80% treino, 20% validação)
    tamanho_treino = int(0.8 * len(dataset))
    tamanho_val = len(dataset) - tamanho_treino
    treino_ds, val_ds = random_split(dataset, [tamanho_treino, tamanho_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(treino_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print("🧠 Construindo modelo CNN...")
    model = CNNUsoSolo()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_sem_melhora = 0

    print("🚀 Iniciando treinamento com early stopping...")
    for epoch in range(epochs):
        model.train()
        loss_acum = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            loss_acum += loss.item()

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_out = model(val_x)
                val_loss += criterion(val_out, val_y).item()

        val_loss /= len(val_loader)

        print(f"🔁 Epoch {epoch+1}/{epochs} - Treino Loss: {loss_acum:.4f} | Val Loss: {val_loss:.4f}")

        # Verifica melhoria
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "resultados/modelo_rna.pth")
            print("💾 Modelo salvo (melhor até agora).")
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= early_stop_patience:
                print("⏹️ Early stopping ativado.")
                break

    print("✅ Treinamento concluído.")

    print("🧪 Avaliando modelo final salvo...")
    model.load_state_dict(torch.load("resultados/modelo_rna.pth"))
    model.eval()

    val_loader = DataLoader(val_ds, batch_size=batch_size)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            out = model(batch_x)
            preds = torch.argmax(out, dim=1)
            y_true.extend(batch_y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    os.makedirs("resultados", exist_ok=True)

    # Relatório e matriz de confusão
    print("📊 Gerando relatório...")
    with open("resultados/rna_relatorio.txt", "w") as f:
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nMatriz de Confusão:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Vegetação", "Não vegetação", "Água"],
                yticklabels=["Vegetação", "Não vegetação", "Água"])
    plt.title("Matriz de Confusão - RNA")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig("resultados/matriz_confusao_rna.png")
    print("📈 Matriz de confusão salva em resultados/matriz_confusao_rna.png")
