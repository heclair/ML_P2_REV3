import os
import numpy as np
import rasterio
from tqdm import tqdm

# Bandas a serem usadas (excluindo B8, B9, B10)
BANDAS_VALIDAS = [
    "B01", "B02", "B03", "B04", "B05",
    "B06", "B07", "B8A", "B11", "B12"
]

# Mapeamento dos rótulos da máscara SCL
MAPA_CLASSES = {
    4: 0,  # Vegetação
    5: 1,  # Não vegetação
    6: 2,  # Água
    # Todos os outros → 3 (não definida)
}

def carregar_banda(caminho):
    with rasterio.open(caminho) as src:
        return src.read(1)

def normalizar(banda):
    return (banda - np.min(banda)) / (np.max(banda) - np.min(banda) + 1e-6)

def carregar_dados(path_img="./img_sentinel/"):
    bandas = []
    shape_ref = None

    print("📥 Carregando bandas:")
    for codigo in BANDAS_VALIDAS:
        filename = [f for f in os.listdir(path_img) if f"_{codigo}_" in f]
        if not filename:
            raise FileNotFoundError(f"Banda {codigo} não encontrada em {path_img}")
        filepath = os.path.join(path_img, filename[0])
        print(f"  - {codigo}: {filename[0]}")
        banda = carregar_banda(filepath)
        bandas.append(normalizar(banda))
        if shape_ref is None:
            shape_ref = banda.shape

    # Empilha bandas como (altura, largura, n_bandas)
    img_stack = np.stack(bandas, axis=-1)

    # Carrega a máscara SCL
    mask_file = [f for f in os.listdir(path_img) if "_SCL_" in f][0]
    path_mask = os.path.join(path_img, mask_file)
    print(f"📥 Carregando máscara: {mask_file}")
    scl = carregar_banda(path_mask)

    # Confere alinhamento
    if scl.shape != shape_ref:
        raise ValueError("A máscara SCL não tem o mesmo shape das bandas.")

    # Pré-aloca X e y
    altura, largura, n_bandas = img_stack.shape
    X, y = [], []

    print("🧮 Gerando vetores espectrais...")
    for i in tqdm(range(altura)):
        for j in range(largura):
            classe_raw = scl[i, j]
            classe = MAPA_CLASSES.get(classe_raw, 3)  # default = não definida
            if classe == 3:
                continue
            pixel = img_stack[i, j, :]
            X.append(pixel)
            y.append(classe)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\n✅ Total de amostras válidas: {len(X)}")
    print(f"🔢 Shape final de X: {X.shape}")
    print(f"🔢 Shape final de y: {y.shape}")

    # Limita a quantidade de amostras (ex: 100 mil) para evitar estouro de memória
    max_amostras = 100_000
    if len(X) > max_amostras:
        print(f"🔻 Reduzindo amostras para {max_amostras} para economizar memória...")
        indices = np.random.choice(len(X), max_amostras, replace=False)
        X = X[indices]
        y = y[indices]

    return X, y
