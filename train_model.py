
    for class_name in classes:
        folder = DATA_DIR / class_name
        if not folder.exists():
            raise FileNotFoundError(
                f"找不到資料夾：{folder}. 請建立 data/{class_name}/ 並放入圖片。"
            )

        files = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
        if not files:
            raise ValueError(
                f"{folder} 裡沒有圖片。請放入已正確標記的 {class_name} 圖片。"
            )

        for file_path in files:
            try:
                with Image.open(file_path) as image:
                    X.append(image_to_features(image))
                    y.append(class_name)
            except (UnidentifiedImageError, OSError) as exc:
                print(f"略過無法讀取的圖片：{file_path} ({exc})")

    if not X:
        raise ValueError("沒有讀取到有效圖片。")
    return np.asarray(X), np.asarray(y), classes

def main():
    X, y, classes = load_dataset()
    counts = {name: int(np.sum(y == name)) for name in classes}
    print("圖片數量：", counts)

    if min(counts.values()) < 2:
        raise ValueError("每個類別至少需要 2 張圖片才能分割資料；實際展示建議準備更多。")

    # Reserve a test set. Stratification keeps both labels represented where possible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", probability=True, class_weight="balanced",
                    random_state=42)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n=== 測試結果（僅代表目前資料集）===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=classes, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; order=Normal, Defective):")
    print(confusion_matrix(y_test, predictions, labels=classes))

    # Store metadata alongside the trained model.
    bundle = {
        "model": model,
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "feature_method": "HOG",
        "model_method": "Linear SVM",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n模型已儲存：{MODEL_PATH}")
    print("接著執行：streamlit run app.py")

if __name__ == "__main__":
    main()
