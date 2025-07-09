import os
import torch
import numpy as np
import rasterio
from tqdm import tqdm
import time
from collections import Counter

# Bandas válidas (exceto B8, B9, B10)
BANDAS_VALIDAS = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B8A", "B11", "B12"
]

MAPA_CLASSES = {
    4: 0,  # Vegetação
    5: 1,  # Não vegetação
    6: 2   # Água
}

PATCH_SIZE = 32
HALF = PATCH_SIZE // 2

def carregar_banda(caminho):
    with rasterio.open(caminho) as src:
        return src.read(1)

def normalizar(banda):
    return (banda - np.min(banda)) / (np.max(banda) - np.min(banda) + 1e-6)

def gerar_patches(path_img="./img_sentinel/", salvar_em="./resultados/", max_amostras=100_000):
    inicio = time.time()
    bandas = []
    shape_ref = None

    print("📥 Carregando bandas:")
    for codigo in BANDAS_VALIDAS:
        filename = [f for f in os.listdir(path_img) if f"_{codigo}_" in f]
        if not filename:
            raise FileNotFoundError(f"Banda {codigo} não encontrada.")
        filepath = os.path.join(path_img, filename[0])
        print(f"  - {codigo}: {filename[0]}")
        banda = carregar_banda(filepath)
        bandas.append(normalizar(banda))
        if shape_ref is None:
            shape_ref = banda.shape

    imagem = np.stack(bandas, axis=0)

    mask_file = [f for f in os.listdir(path_img) if "_SCL_" in f][0]
    path_mask = os.path.join(path_img, mask_file)
    print(f"📥 Carregando máscara: {mask_file}")
    mascara = carregar_banda(path_mask)

    if mascara.shape != shape_ref:
        raise ValueError("Máscara com shape diferente das bandas.")

    altura, largura = mascara.shape

    print("🧮 Gerando patches...")
    patches = []
    labels = []

    dist_classes = Counter()

    for i in tqdm(range(HALF, altura - HALF)):
        for j in range(HALF, largura - HALF):
            classe_raw = mascara[i, j]
            if classe_raw not in MAPA_CLASSES:
                continue

            patch = imagem[:, i - HALF:i + HALF, j - HALF:j + HALF]
            if patch.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
                continue

            patches.append(patch)
            labels.append(MAPA_CLASSES[classe_raw])
            dist_classes[MAPA_CLASSES[classe_raw]] += 1

    print("\n🔎 Distribuição das classes:")
    for classe, count in dist_classes.items():
        nome = ["Vegetação", "Não vegetação", "Água"][classe]
        print(f"Classe {classe} - {nome}: {count} amostras")

    print(f"\n⚠️ Reduzindo para {max_amostras} amostras aleatórias para economizar memória...")
    indices = np.random.choice(len(patches), max_amostras, replace=False)

    X_reduzido = [patches[i] for i in indices]
    y_reduzido = [labels[i] for i in indices]

    X_tensor = torch.tensor(X_reduzido, dtype=torch.float32)
    y_tensor = torch.tensor(y_reduzido, dtype=torch.long)

    os.makedirs(salvar_em, exist_ok=True)
    torch.save(X_tensor, os.path.join(salvar_em, "X_rna.pt"))
    torch.save(y_tensor, os.path.join(salvar_em, "y_rna.pt"))

    fim = time.time()
    print(f"\n✅ {len(X_tensor)} patches salvos em {salvar_em}")
    print(f"⏱️ Tempo total: {fim - inicio:.2f} segundos")
