import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



data_dir = 'flowers/'  


datagen = ImageDataGenerator(
    rescale=1./255,        
    validation_split=0.2   
)


train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),  
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax") 
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_generator, validation_data=val_generator, epochs=10)
model.save("flower_classifier.h5") 
loss, acc = model.evaluate(train_generator)
print(f"Accuracy: {acc:.4f}")
pred_probs = model.predict(train_generator)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = train_generator.classes
cm = confusion_matrix(true_labels, pred_labels)
class_names = list(train_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
