import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (GlobalAveragePooling2D, Input, Dense, 
                                    Dropout, BatchNormalization, Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall, F1Score
from tensorflow.keras.losses import BinaryFocalCrossentropy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, precision_score, 
                            recall_score, f1_score, roc_auc_score)

# ========================= IMAGE PROCESSING FUNCTIONS =========================

def load_images_and_labels(img_dir, img_size=(300, 300)):  # EfficientNetB3 input size
    """Load and enhance X-ray images from directory"""
    images, img_names = [], []
    
    for filename in os.listdir(img_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(img_dir, filename)
                
                # Load and preprocess image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load {filename}")
                    continue
                
                # Resize to standard dimensions (300x300 for EfficientNetB3)
                img = cv2.resize(img, img_size)
                
                # Enhance contrast with CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                
                # Normalize to [0,1]
                img = img / 255.0
                
                # Convert to 3-channel by duplicating grayscale
                img_rgb = np.stack([img, img, img], axis=2)
                
                # Apply EfficientNet specific preprocessing
                img_rgb = preprocess_input_efficientnet(img_rgb)
                
                images.append(img_rgb)
                img_names.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return np.array(images, dtype=np.float16), img_names

def preprocess_input_efficientnet(img):
    """Apply EfficientNet specific preprocessing"""
    # EfficientNet uses different normalization than VGG16
    # Use TensorFlow's built-in preprocessing function
    return tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))

# ========================= LABEL PROCESSING FUNCTIONS =========================

def preprocess_labels(csv_path):
    """Process multi-label data from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Clean class names - standardize format
    df['class_name'] = df['class_name'].apply(
        lambda x: [label.strip() for label in x.split('|')] 
                 if '|' in str(x) else [str(x).strip()]
    )
    
    # Convert to binary format using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['class_name'])
    
    # Get clean class names
    clean_classes = [cls.strip() for cls in mlb.classes_]
    
    print(f"Processed {len(df)} images with {len(clean_classes)} classes")
    print(f"Classes: {clean_classes}")
    
    return labels, clean_classes, df['image_id'].values

# ========================= FEATURE EXTRACTION =========================

def extract_efficientnet_features(images, batch_size=16):  # smaller batch size for EfficientNet
    """Extract features using pre-trained EfficientNetB3 with partial fine-tuning"""
    # Create base model with ImageNet weights
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Freeze early layers, make later ones trainable
    # EfficientNet has more blocks than VGG, adjust freezing accordingly
    # Generally, freezing about 70% of the network is a good starting point
    total_layers = len(base_model.layers)
    freeze_layers = int(total_layers * 0.7)
    
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True
    
    # Extract features in batches
    features = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in range(0, len(images), batch_size):
        if i % 10 == 0:
            print(f"Processing batch {i//batch_size + 1}/{num_batches}")
        batch = images[i:i+batch_size]
        batch_features = model.predict(batch, verbose=0)
        features.append(batch_features)
    
    return np.vstack(features)

# ========================= MODEL BUILDING FUNCTIONS =========================

def calc_class_weights(labels):
    """Calculate class weights inversely proportional to frequency"""
    # Count samples per class
    class_counts = np.sum(labels, axis=0)
    
    # Calculate weights
    n_samples = labels.shape[0]
    n_classes = labels.shape[1]
    
    # Handle zero counts
    class_counts = np.maximum(class_counts, 1)
    
    # Use inverse frequency with scaling
    weights = n_samples / (n_classes * class_counts)
    
    # Normalize to average of 1.0
    weights = weights / np.mean(weights)
    
    # Cap extreme weights to avoid training instability
    weights = np.minimum(weights, 10.0)
    
    # Convert to dictionary
    class_weights = {i: float(w) for i, w in enumerate(weights)}
    
    print("Class weights summary:")
    for i, w in class_weights.items():
        print(f"Class {i}: {w:.2f}")
    
    return class_weights

def create_improved_model(input_shape, num_classes):
    """Create model with residual connections optimized for EfficientNet features"""
    image_input = Input(shape=(input_shape,), name='image_features')
    
    # First block - EfficientNet features are higher dimension than VGG
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0005))(image_input)
    x = BatchNormalization()(x)
    x1 = Dropout(0.3)(x)  # Save for residual connection
    
    # Second block
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x1)
    x = BatchNormalization()(x)
    x2 = Dropout(0.3)(x)  # Save for 2nd residual connection
    
    # Third block with first residual connection
    x = Dense(512, activation='relu')(x2)
    x = BatchNormalization()(x)
    x = Add()([x, x2])  # Add residual connection
    x = Dropout(0.3)(x)
    
    # Fourth block with second residual connection
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Add()([x, x1])  # Add residual connection to first block
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=image_input, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.0003)  # Slightly lower learning rate for EfficientNet
    model.compile(
        optimizer=optimizer,
        loss=BinaryFocalCrossentropy(gamma=2.0),  # Using built-in focal loss
        metrics=['accuracy', 
                AUC(multi_label=True), 
                F1Score(threshold=0.5, average='weighted'),
                Precision(), 
                Recall()]
    )
    
    return model

# ========================= EVALUATION FUNCTIONS =========================

def optimize_thresholds(model, val_features, val_labels):
    """Find optimal threshold for each class to maximize F1 score"""
    predictions = model.predict(val_features)
    optimal_thresholds = []
    
    for i in range(val_labels.shape[1]):
        best_f1 = 0
        best_threshold = 0.5
        
        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred_i = (predictions[:, i] > threshold).astype(int)
            f1 = f1_score(val_labels[:, i], pred_i, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"Class {i}: optimal threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")
    
    return optimal_thresholds

def evaluate_model(model, test_features, test_labels, label_names, thresholds=None):
    """Evaluate model with detailed metrics per class"""
    # Get predictions
    predictions = model.predict(test_features)
    
    # Apply thresholds (default 0.5 if not specified)
    if thresholds is None:
        thresholds = [0.5] * len(label_names)
        
    pred_binary = np.zeros_like(predictions, dtype=int)
    for i in range(predictions.shape[1]):
        pred_binary[:, i] = (predictions[:, i] > thresholds[i]).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_binary, target_names=label_names))
    
    # Per-class metrics
    print("\nPer-class metrics:")
    for i, class_name in enumerate(label_names):
        precision = precision_score(test_labels[:, i], pred_binary[:, i], zero_division=0)
        recall = recall_score(test_labels[:, i], pred_binary[:, i], zero_division=0)
        f1 = f1_score(test_labels[:, i], pred_binary[:, i], zero_division=0)
        
        try:
            auc = roc_auc_score(test_labels[:, i], predictions[:, i])
        except:
            auc = float('nan')
            
        print(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Overall metrics
    print("\nOverall metrics:")
    weighted_precision = precision_score(test_labels, pred_binary, average='weighted', zero_division=0)
    weighted_recall = recall_score(test_labels, pred_binary, average='weighted', zero_division=0)
    weighted_f1 = f1_score(test_labels, pred_binary, average='weighted', zero_division=0)
    
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall: {weighted_recall:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    return pred_binary, predictions

# ========================= MAIN FUNCTION =========================

def main():
    # Define paths
    img_dir = r"C:\Users\Ishaan\Desktop\Clg\4th Sem\Practicum\Sample\18000\Groupedlabel1 wala\images"
    csv_path = r"C:\Users\Ishaan\Desktop\Clg\4th Sem\Practicum\Sample\18000\Groupedlabel1 wala\grouped_labels (1).csv"
    
    # 1. Process labels
    print("Processing labels...")
    labels, label_names, image_ids = preprocess_labels(csv_path)
    
    # 2. Load and preprocess images
    print("Processing images...")
    images, img_names = load_images_and_labels(img_dir)
    
    # 3. Match labels with images
    print("Aligning datasets...")
    image_to_index = {os.path.splitext(os.path.basename(name))[0]: i for i, name in enumerate(img_names)}
    
    valid_indices = []
    valid_labels = []
    
    for i, image_id in enumerate(image_ids):
        if image_id in image_to_index:
            valid_indices.append(image_to_index[image_id])
            valid_labels.append(labels[i])
    
    aligned_images = images[valid_indices]
    aligned_labels = np.array(valid_labels)
    
    print(f"Successfully aligned {len(aligned_images)} images with labels")
    
    # 4. Split into train/validation/test sets
    print("Splitting data...")
    train_indices, temp_indices = train_test_split(
        range(len(aligned_images)), 
        test_size=0.3,
        random_state=42,
        stratify=np.argmax(aligned_labels, axis=1) if aligned_labels.shape[1] > 1 else aligned_labels
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=42,
        stratify=np.argmax(aligned_labels[temp_indices], axis=1) if aligned_labels.shape[1] > 1 else aligned_labels[temp_indices]
    )
    
    # Create dataset splits
    train_images = aligned_images[train_indices]
    train_labels = aligned_labels[train_indices]
    val_images = aligned_images[val_indices]
    val_labels = aligned_labels[val_indices]
    test_images = aligned_images[test_indices]
    test_labels = aligned_labels[test_indices]
    
    # 5. Extract features using EfficientNetB3
    print("Extracting features using pre-trained EfficientNetB3...")
    train_features = extract_efficientnet_features(train_images)
    val_features = extract_efficientnet_features(val_images)
    test_features = extract_efficientnet_features(test_images)
    
    # 6. Calculate class weights for handling imbalance
    class_weights = calc_class_weights(train_labels)
    
    # 7. Create and train model
    print("Creating and training model...")
    model = create_improved_model(
        input_shape=train_features.shape[1],
        num_classes=train_labels.shape[1]
    )
    
    # Define callbacks with more patience for EfficientNet
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ]
    
    # Train with class weights
    history = model.fit(
        train_features, 
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=100,  # More epochs for EfficientNet with early stopping
        batch_size=16,  # Smaller batch size for EfficientNet
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # 8. Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1_score'], label='Train F1')
    plt.plot(history.history['val_f1_score'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('EfficientNet_training_history.png')
    plt.show()
    
    # 9. Optimize thresholds per class
    print("Optimizing classification thresholds...")
    optimal_thresholds = optimize_thresholds(model, val_features, val_labels)
    
    # 10. Final evaluation
    print("Evaluating model on test set...")
    test_results = model.evaluate(test_features, test_labels)
    print(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    
    # Detailed evaluation with optimized thresholds
    pred_binary, pred_proba = evaluate_model(
        model, test_features, test_labels, label_names, 
        thresholds=optimal_thresholds
    )
    
    # 11. Save the model
    model.save('efficientnetb3_xray_classifier.h5')
    print("Model saved successfully")

if __name__ == "__main__":
    main()