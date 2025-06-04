import numpy as np
import pandas as pd
import os
import cv2 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet121  # Changed from VGG16 to DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall, F1Score
import tensorflow.keras.backend as K

def load_images_and_labels(img_dir, img_size=(224, 224)):
    """Preprocess X-ray images with enhancement techniques"""
    
    images, img_names = [], []
    
    for filename in os.listdir(img_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img_path = os.path.join(img_dir, filename)
                
                # Load image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load {filename}")
                    continue
                
                # Resize image to expected dimensions
                img = cv2.resize(img, img_size)
                
                # Apply CLAHE for contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                
                # Normalize to [0,1]
                img = img / 255.0
                
                # Convert to RGB (3 channel) by duplicating grayscale
                img_rgb = np.stack([img, img, img], axis=2)
                
                # Apply ImageNet normalization
                img_rgb = normalize_imagenet(img_rgb)
                
                images.append(img_rgb)
                img_names.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return np.array(images, dtype=np.float16), img_names

def normalize_imagenet(img):
    """Apply ImageNet normalization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std

def preprocess_labels(csv_path):
    """Process only the labels from the CSV file without metadata"""
    df = pd.read_csv(csv_path)
    
    # Clean and normalize class names - strip spaces and standardize
    df['class_name'] = df['class_name'].apply(
        lambda x: [label.strip() for label in x.split('|')] if '|' in str(x) else [str(x).strip()]
    )
    
    # Apply MultiLabelBinarizer with clean labels
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(df['class_name'])
    
    # Get clean class names without duplicate spaces
    clean_classes = [cls.strip() for cls in mlb.classes_]
    
    # Create DataFrame with binary labels
    label_df = pd.DataFrame(labels, columns=clean_classes)
    
    # Combine original data with binary labels
    result_df = pd.concat([df, label_df], axis=1)
    
    # Drop the original class_name column
    result_df.drop(columns=['class_name'], inplace=True)
    
    print(f"Processed labels for {len(df)} images with {len(clean_classes)} unique classes")
    print(f"Label names: {clean_classes}")
    
    return labels, clean_classes, df['image_id'].values

def extract_densenet_features(images, batch_size=32):
    """Extract features using pre-trained DenseNet121 with partial fine-tuning"""
    # Create base model with pre-trained ImageNet weights
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Calculate proportion of layers to freeze (first 70%)
    freeze_layers = int(len(base_model.layers) * 0.7)
    
    # Freeze early layers
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False
    # Make later layers trainable for fine-tuning
    for layer in base_model.layers[freeze_layers:]:
        layer.trainable = True
    
    # Extract features in batches (transfer learning approach)
    features = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    
    for i in range(0, len(images), batch_size):
        if i % 10 == 0:
            print(f"Processing batch {i//batch_size + 1}/{num_batches}")
        
        batch = images[i:i+batch_size]
        batch_features = model.predict(batch, verbose=0)
        features.append(batch_features)
    
    return np.vstack(features)

# 1. Implement focal loss for class imbalance
def focal_loss(gamma=2.0, alpha=None):
    def focal_loss_fixed(y_true, y_pred):
        # Clip for numerical stability
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal weight
        focal_weight = K.pow(1 - y_pred, gamma) * y_true + K.pow(y_pred, gamma) * (1 - y_true)
        
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Apply class weights if provided
        if alpha is not None:
            # Convert alpha dict to tensor that matches y_true shape
            alpha_tensor = tf.zeros_like(y_true)
            for cls_idx, weight in alpha.items():
                # Add weight for this class at the appropriate position
                alpha_tensor = alpha_tensor + tf.cast(
                    tf.equal(tf.range(y_true.shape[1]), cls_idx), 
                    dtype=tf.float32
                ) * weight
                
            # Apply per-class weighting
            cross_entropy = cross_entropy * alpha_tensor
        
        return K.mean(focal_weight * cross_entropy)
    
    return focal_loss_fixed

def calc_class_weights(labels):
    """
    Calculate class weights inversely proportional to class frequency
    
    Parameters:
    - labels: binary matrix of labels (n_samples, n_classes)
    
    Returns:
    - Dictionary mapping class indices to weights
    """
    # Count positive examples per class
    class_counts = np.sum(labels, axis=0)
    
    # Calculate weights inversely proportional to frequency
    n_samples = labels.shape[0]
    n_classes = labels.shape[1]
    
    # Handle zero counts
    class_counts = np.maximum(class_counts, 1)
    
    # Use inverse frequency with scaling
    weights = n_samples / (n_classes * class_counts)
    
    # Normalize weights to average at 1.0
    weights = weights / np.mean(weights)
    
    # Cap extremely high weights to avoid training instability
    weights = np.minimum(weights, 10.0)
    
    # Convert to dictionary
    class_weights = {i: float(w) for i, w in enumerate(weights)}
    
    print("Class weights calculated:")
    for i, w in class_weights.items():
        print(f"Class {i}: {w:.2f}")
    
    return class_weights

# 3. Create improved model with residual connections
def create_improved_model(image_feature_shape, num_classes):
    """Create a more robust model with residual connections"""
    image_input = Input(shape=(image_feature_shape,), name='image_features')
    
    # First block
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(image_input)
    x = BatchNormalization()(x)
    x1 = Dropout(0.3)(x)
    
    # Second block
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third block with residual connection
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x1])  # Residual connection
    
    # Final classification layers
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=image_input, outputs=outputs)
    
    # Compile with optimal hyperparameters
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),
        metrics=['accuracy', 
                 AUC(multi_label=True), 
                 F1Score(threshold=0.5, average='weighted'),
                 Precision(), 
                 Recall()]
    )
    
    return model

# 5. Threshold optimization per class (to be used after training)
def optimize_thresholds(model, val_features, val_labels):
    """Find optimal threshold for each class to maximize F1 score"""
    predictions = model.predict(val_features)
    optimal_thresholds = []
    
    for i in range(val_labels.shape[1]):
        # Try different thresholds
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred_i = (predictions[:, i] > threshold).astype(int)
            f1 = f1_score(val_labels[:, i], pred_i, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        print(f"Class {i}: optimal threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")
    
    return optimal_thresholds

def evaluate_model(model, test_image_features, test_labels, label_names, thresholds=None):
    """Evaluate the model and print detailed metrics"""
    # Get predictions
    predictions = model.predict(test_image_features)
    
    # Apply thresholds (default 0.5 if not specified)
    if thresholds is None:
        thresholds = [0.5] * len(label_names)
        
    pred_binary = np.zeros_like(predictions, dtype=int)
    for i in range(predictions.shape[1]):
        pred_binary[:, i] = (predictions[:, i] > thresholds[i]).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, pred_binary, target_names=label_names))
    
    # Calculate and print per-class metrics
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

def genetic_algorithm_feature_selection(data, labels, pop_size=30, generations=20, mutation_rate=0.3):
    feature_size = data.shape[1]
    population = np.random.randint(0, 2, (pop_size, feature_size), dtype=np.uint8)
    
    print(f"Running genetic algorithm for feature selection on {feature_size} features")
    print(f"Population size: {pop_size}, Generations: {generations}")
    
    best_fitness_history = []
    
    for gen in range(generations):
        fitness_scores = evaluate_fitness(population, data, labels)
        best_fitness = np.max(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        best_individual = population[np.argmax(fitness_scores)]
        num_selected = np.sum(best_individual)
        
        print(f"Generation {gen+1}/{generations}: Best fitness = {best_fitness:.4f}, Selected features: {num_selected}/{feature_size}")
        
        parents = tournament_selection(population, fitness_scores)
        offspring = crossover(parents)
        offspring = mutate(offspring, mutation_rate)
        population = np.vstack((parents[:pop_size//2], offspring[:pop_size//2]))
    
    # Get the best individual from the final population
    final_fitness_scores = evaluate_fitness(population, data, labels)
    best_features = population[np.argmax(final_fitness_scores)]
    
    # Ensure at least one feature is selected
    if np.sum(best_features) == 0:
        best_features[np.random.randint(feature_size)] = 1
    
    selected_count = np.sum(best_features)
    print(f"Genetic algorithm completed. Selected {selected_count}/{feature_size} features")
    
    # Plot the fitness history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations+1), best_fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Genetic Algorithm Progress')
    plt.grid(True)
    plt.show()
    
    return best_features

def evaluate_fitness(population, data, labels):
    scores = np.zeros(population.shape[0])
    
    for i, individual in enumerate(population):
        selected_features = data[:, individual.astype(bool)]
        selected_count = np.sum(individual)
        
        # Skip empty feature sets
        if selected_features.shape[1] == 0:
            continue
            
        # Apply penalty for too many or too few features
        # This encourages more efficient feature selection
        if selected_count < 3:
            # Strong penalty for too few features
            feature_penalty = -0.5
        else:
            # Mild penalty proportional to feature count
            # Encourages more compact feature sets without being too aggressive
            feature_penalty = -0.01 * selected_count / data.shape[1]
        
        # Split data with stratification to preserve class distribution
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                selected_features, labels, test_size=0.2, 
                random_state=42, stratify=np.argmax(labels, axis=1) if labels.shape[1] > 1 else labels
            )
        except ValueError:
            # If stratification fails, proceed without it
            X_train, X_val, y_train, y_val = train_test_split(
                selected_features, labels, test_size=0.2, random_state=42
            )
        
        # Build a model with dynamic architecture based on feature count
        hidden_units = min(128, max(32, 2 * selected_features.shape[1]))
        
        model = Sequential([
            Dense(hidden_units, activation='relu', input_shape=(selected_features.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(labels.shape[1], activation='sigmoid')
        ])
        
        # Compile with appropriate loss based on label distribution
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping to prevent overfitting and speed up evaluation
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        # Train the model with more appropriate batch size and fewer epochs
        model.fit(
            X_train, y_train, 
            epochs=5, 
            batch_size=min(32, len(X_train)), 
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Get predictions
        val_preds = model.predict(X_val, verbose=0)
        
        # Calculate multiple performance metrics
        val_preds_binary = (val_preds > 0.5).astype(int)
        
        # F1 score (samples accounts for multilabel)
        try:
            f1 = f1_score(y_val, val_preds_binary, average='samples', zero_division=0)
        except:
            f1 = 0
            
        # AUC for multilabel
        try:
            # For multilabel, use macro average
            if labels.shape[1] > 1:
                auc = roc_auc_score(y_val, val_preds, average='macro')
            else:
                auc = roc_auc_score(y_val, val_preds)
        except ValueError:
            auc = 0.5  # Default if AUC fails
            
        # Get accuracy
        accuracy = np.mean(np.all(val_preds_binary == y_val, axis=1))
        
        # Combine metrics with appropriate weights
        # Balance between accuracy, F1, AUC, and feature penalty
        combined_score = (0.4 * f1 + 0.3 * auc + 0.3 * accuracy) + feature_penalty
        
        # Store the final score
        scores[i] = combined_score
    
    return scores

def tournament_selection(population, fitness_scores, tournament_size=4):
    selected = []
    for _ in range(len(population) // 2):
        competitors = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = competitors[np.argmax(fitness_scores[competitors])]
        selected.append(population[best_idx])
    return np.array(selected)

def crossover(parents):
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        crossover_point = np.random.randint(1, len(parent1) - 1)
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        offspring.extend([child1, child2])
    
    return np.array(offspring)

def mutate(offspring, mutation_rate=0.1): #Bit Flip Mutation
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_id = np.random.randint(len(offspring[i]))
            offspring[i][mutation_id] = 1 - offspring[i][mutation_id]
    return offspring

def main():
    # Define paths
    img_dir = r"C:\Users\Ishaan\Desktop\Clg\4th Sem\Practicum\Sample\18000\Groupedlabel1 wala\images"
    csv_path = r"C:\Users\Ishaan\Desktop\Clg\4th Sem\Practicum\Sample\18000\Groupedlabel1 wala\grouped_labels (1).csv"

    # Process labels (without metadata)
    print("Processing labels...")
    labels, label_names, image_ids = preprocess_labels(csv_path)
    
    # Load and preprocess images
    print("Processing images...")
    images, img_names = load_images_and_labels(img_dir)
    
    # Match labels with images
    print("Aligning datasets...")
    image_to_index = {os.path.splitext(os.path.basename(name))[0]: i for i, name in enumerate(img_names)}
    
    valid_indices = []
    valid_labels = []
    
    # For each image_id in the CSV, find the corresponding image in our loaded images
    for i, image_id in enumerate(image_ids):
        if image_id in image_to_index:
            valid_indices.append(image_to_index[image_id])
            valid_labels.append(labels[i])
    
    # Get the aligned images and labels
    aligned_images = images[valid_indices]
    aligned_labels = np.array(valid_labels)
    
    print(f"Successfully aligned {len(aligned_images)} images with labels")
    
    # Now split the dataset into train, validation, and test sets
    print("Splitting data into train/validation/test sets...")
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
    
    # Create train/val/test sets for images and labels
    train_images = aligned_images[train_indices]
    train_labels = aligned_labels[train_indices]
    val_images = aligned_images[val_indices]
    val_labels = aligned_labels[val_indices]
    test_images = aligned_images[test_indices]
    test_labels = aligned_labels[test_indices]
    
    # Extract features using DenseNet121 instead of VGG16
    print("Extracting features using pre-trained DenseNet121...")
    train_image_features = extract_densenet_features(train_images)
    val_image_features = extract_densenet_features(val_images)
    test_image_features = extract_densenet_features(test_images)
    
    # Apply genetic algorithm for feature selection
    print("Applying genetic algorithm for feature selection...")
    selected_features = genetic_algorithm_feature_selection(
        train_image_features, 
        train_labels,
        pop_size=30, 
        generations=15,
        mutation_rate=0.2
    )
    
    # Filter features based on GA selection
    print(f"Selected {np.sum(selected_features)} out of {train_image_features.shape[1]} features")
    train_image_features_selected = train_image_features[:, selected_features.astype(bool)]
    val_image_features_selected = val_image_features[:, selected_features.astype(bool)]
    test_image_features_selected = test_image_features[:, selected_features.astype(bool)]
    
    # Calculate class weights for remaining imbalance
    class_weights = calc_class_weights(train_labels)
    
    # Create and train improved model
    print("Creating and training improved model with selected features...")
    model = create_improved_model(
        image_feature_shape=train_image_features_selected.shape[1],
        num_classes=train_labels.shape[1]
    )
    
    # Define improved callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ]
    
    # Train with class weights to address any remaining imbalance
    history = model.fit(
        train_image_features_selected, 
        train_labels,
        validation_data=(val_image_features_selected, val_labels),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights  # Apply class weights during training
    )
    
    # Plot training history
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
    plt.show()
    
    # Optimize thresholds per class
    print("Optimizing classification thresholds...")
    optimal_thresholds = optimize_thresholds(model, val_image_features_selected, val_labels)
    
    # Final evaluation on test set with optimized thresholds
    print("Evaluating model on test set...")
    test_results = model.evaluate(test_image_features_selected, test_labels)
    print(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    
    # Detailed evaluation with optimized thresholds
    pred_binary, pred_proba = evaluate_model(
        model, test_image_features_selected, test_labels, label_names, 
        thresholds=optimal_thresholds
    )

if __name__ == "__main__":
    main()