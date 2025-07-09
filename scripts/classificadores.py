from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def salvar_resultados(nome_modelo, y_true, y_pred):
    os.makedirs("resultados", exist_ok=True)

    # Salva o relatório textual
    with open(f"resultados/{nome_modelo}_relatorio.txt", "w") as f:
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nMatriz de Confusão:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    # Gera e salva a imagem da matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Vegetação", "Não Vegetação", "Água"],
                yticklabels=["Vegetação", "Não Vegetação", "Água"])
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"resultados/{nome_modelo}_matriz_confusao.png")
    plt.close()

def treinar_random_forest(X, y):
    print("🏁 Iniciando classificação com Random Forest...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    print("🌲 Treinando Random Forest...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    salvar_resultados("random_forest", y_test, y_pred)
    joblib.dump(clf, "resultados/random_forest_model.pkl")
    print("✅ Random Forest finalizado e resultados salvos em /resultados")

def treinar_svm(X, y):
    print("🏁 Iniciando classificação com SVM...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = SVC(kernel="rbf", C=1.0)
    print("🧠 Treinando SVM...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    salvar_resultados("svm", y_test, y_pred)
    joblib.dump(clf, "resultados/svm_model.pkl")
    print("✅ SVM finalizado e resultados salvos em /resultados")
