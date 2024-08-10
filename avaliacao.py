from sklearn.metrics import classification_report, confusion_matrix

# Avaliando o modelo
y_pred = svm.predict(X_test)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
