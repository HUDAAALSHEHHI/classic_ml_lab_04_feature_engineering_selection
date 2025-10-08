# Quick overview of the dataset
print("Dataset Shape:", X.shape)
print("Selected Features:", list(selected_features))
print("Top Features by Score:\n", feat_importance.head())

# Display final accuracy and evaluation metrics
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("Final Test F1-macro:", f1_score(y_test, y_pred, average='macro'))
