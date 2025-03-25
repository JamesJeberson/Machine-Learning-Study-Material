### **Early Stopping in Neural Networks (NN)**
Early stopping is a **regularization technique** used to **prevent overfitting** by stopping training when the model’s performance stops improving on the validation set.  

---
## **🔹 How Early Stopping Works?**
1. **Monitor a Metric** – Typically, **validation loss (`val_loss`)** or **validation accuracy (`val_accuracy`)**.  
2. **Set a Patience Level** – The number of epochs to wait before stopping if no improvement is seen.  
3. **Restore Best Weights** – Optionally, restore the best weights when stopping.

---
## **🔹 Early Stopping in TensorFlow/Keras**
Keras provides the **`EarlyStopping`** callback to implement early stopping.

### **✅ Example: Early Stopping in a Simple NN**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Create a simple Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),  # 10 input features
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=5,           # Wait for 5 epochs before stopping if no improvement
    restore_best_weights=True,  # Restore the best model weights
    verbose=1
)

# Train the model with early stopping
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=100,  # Large epoch count, since early stopping will stop earlier
                    batch_size=32,
                    callbacks=[early_stopping])
```

---
## **🔹 Explanation of Parameters**
| Parameter | Description |
|-----------|-------------|
| `monitor='val_loss'` | Metric to monitor (common choices: `val_loss`, `val_accuracy`). |
| `patience=5` | Number of epochs to wait before stopping if no improvement. |
| `restore_best_weights=True` | Restores model to the best weights when stopping. |
| `verbose=1` | Prints a message when early stopping occurs. |

---
## **🔹 Example Output**
```
Epoch 15: early stopping
```
This means training was **stopped at epoch 15** because validation loss stopped improving for **5 consecutive epochs**.

---
## **🔹 When to Use Early Stopping?**
✅ If your model **overfits quickly** → Early stopping helps prevent this.  
✅ When training **deep neural networks** → Saves computation time.  
✅ When dataset size is **small** → Helps generalization.  

---
## **🔹 Best Practices**
- Use **`restore_best_weights=True`** to ensure the model does not overfit before stopping.  
- **Tune `patience`** (too small may stop too early, too large may overfit).  
- Monitor **`val_loss`** instead of `loss` to track generalization ability.  
