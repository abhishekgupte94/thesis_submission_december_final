import torch
from sklearn.model_selection import train_test_split

# Assuming you have your stacked features and labels
X = stack_features(npv_features, art_avdf_features)
y = labels

# Split the data into train_filenames, validation, and inference sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Perform feature selection
X_train_selected, selector = select_features(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the meta-learner
meta_learner = train_meta_learner(X_train_selected, y_train, X_val_selected, y_val,
                                  X_train_selected.shape[1], device, epochs=50, batch_size=32)

# Evaluate the model
val_accuracy = evaluate_meta_learner(meta_learner, X_val_selected, y_val, device)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Test the model
test_accuracy = evaluate_meta_learner(meta_learner, X_test_selected, y_test, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
