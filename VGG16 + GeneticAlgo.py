import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16  # Changed back to VGG16 from DenseNet121
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

def load_images_and_labels(img_dir, img_size=(224, 224)):
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
                
                # Resize to standard dimensions
                img = cv2.resize(img, img_size)
                
                # Enhance contrast with CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                
                # Normalize to [0,1]
                img = img / 255.0
                
                # Convert to 3-channel by duplicating grayscale
                img_rgb = np.stack([img, img, img], axis=2)
                
                # Apply ImageNet normalization
                img_rgb = normalize_imagenet(img_rgb)
                
                images.append(img_rgb)
                img_names.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return np.array(images, dtype=np.float16), img_names

def normalize_imagenet(img):
    """Apply ImageNet normalization with mean and std"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std

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

def extract_vgg_features(images, batch_size=32):
    """Extract features using pre-trained VGG16 with partial fine-tuning"""
    # Create base model with ImageNet weights
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    output = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Freeze early layers, make later ones trainable
    # VGG16 has a simpler sequential structure than DenseNet
    # Freeze first 15 layers (out of 19 in base model)
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
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
    """Create model with residual connections for better gradient flow"""
    image_input = Input(shape=(input_shape,), name='image_features')
    
    # First block
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(image_input)
    x = BatchNormalization()(x)
    x1 = Dropout(0.3)(x)  # Save for residual connection
    
    # Second block
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third block with residual connection
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x1])  # Add residual connection
    
    # Final classification layers
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=image_input, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.0005)
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

# ========================= FEATURE SELECTION - GENETIC ALGORITHM =========================

def genetic_algorithm_feature_selection(data, labels, pop_size=30, generations=20, mutation_rate=0.3):
    """Use genetic algorithm to select optimal features"""
    feature_size = data.shape[1]
    # Initialize random population (1=feature used, 0=feature not used)
    population = np.random.randint(0, 2, (pop_size, feature_size), dtype=np.uint8)
    
    print(f"Running genetic algorithm for feature selection on {feature_size} features")
    print(f"Population size: {pop_size}, Generations: {generations}")
    
    best_fitness_history = []
    
    for gen in range(generations):
        # Evaluate fitness of current population
        fitness_scores = evaluate_fitness(population, data, labels)
        best_fitness = np.max(fitness_scores)
        best_fitness_history.append(best_fitness)
        
        best_individual = population[np.argmax(fitness_scores)]
        num_selected = np.sum(best_individual)
        
        print(f"Generation {gen+1}/{generations}: Best fitness = {best_fitness:.4f}, Selected features: {num_selected}/{feature_size}")
        
        # Selection, crossover and mutation
        parents = tournament_selection(population, fitness_scores)
        offspring = crossover(parents)
        offspring = mutate(offspring, mutation_rate)
        population = np.vstack((parents[:pop_size//2], offspring[:pop_size//2]))
    
    # Get best solution from final population
    final_fitness_scores = evaluate_fitness(population, data, labels)
    best_features = population[np.argmax(final_fitness_scores)]
    
    # Ensure at least one feature is selected
    if np.sum(best_features) == 0:
        best_features[np.random.randint(feature_size)] = 1
    
    selected_count = np.sum(best_features)
    print(f"Genetic algorithm completed. Selected {selected_count}/{feature_size} features")
    
    # Plot fitness progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations+1), best_fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.title('Genetic Algorithm Progress')
    plt.grid(True)
    plt.show()
    
    return best_features

def evaluate_fitness(population, data, labels):
    """Evaluate fitness of each individual in population"""
    scores = np.zeros(population.shape[0])
    
    for i, individual in enumerate(population):
        selected_features = data[:, individual.astype(bool)]
        selected_count = np.sum(individual)
        
        # Skip empty feature sets
        if selected_features.shape[1] == 0:
            continue
            
        # Apply penalty for feature count
        if selected_count < 3:
            # Strong penalty for too few features
            feature_penalty = -0.5
        else:
            # Mild penalty proportional to feature count
            feature_penalty = -0.01 * selected_count / data.shape[1]
        
        # Split data for evaluation
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
        
        # Build simple evaluation model
        hidden_units = min(128, max(32, 2 * selected_features.shape[1]))
        
        model = Sequential([
            Dense(hidden_units, activation='relu', input_shape=(selected_features.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(labels.shape[1], activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping for efficiency
        early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        # Train model
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
        val_preds_binary = (val_preds > 0.5).astype(int)
        
        # Calculate performance metrics
        try:
            f1 = f1_score(y_val, val_preds_binary, average='samples', zero_division=0)
        except:
            f1 = 0
            
        try:
            if labels.shape[1] > 1:
                auc = roc_auc_score(y_val, val_preds, average='macro')
            else:
                auc = roc_auc_score(y_val, val_preds)
        except ValueError:
            auc = 0.5
            
        accuracy = np.mean(np.all(val_preds_binary == y_val, axis=1))
        
        # Combine metrics with weights
        combined_score = (0.4 * f1 + 0.3 * auc + 0.3 * accuracy) + feature_penalty
        scores[i] = combined_score
    
    return scores

def tournament_selection(population, fitness_scores, tournament_size=4):
    """Select individuals using tournament selection"""
    selected = []
    for _ in range(len(population) // 2):
        competitors = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = competitors[np.argmax(fitness_scores[competitors])]
        selected.append(population[best_idx])
    return np.array(selected)

def crossover(parents):
    """Create offspring through single-point crossover"""
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        parent1, parent2 = parents[i], parents[i + 1]
        crossover_point = np.random.randint(1, len(parent1) - 1)
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        offspring.extend([child1, child2])
    
    return np.array(offspring)

def mutate(offspring, mutation_rate=0.1):
    """Apply bit-flip mutation to offspring"""
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mutation_idx = np.random.randint(len(offspring[i]))
            offspring[i][mutation_idx] = 1 - offspring[i][mutation_idx]
    return offspring

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
    
    # 5. Extract features using VGG16 (replaced back from DenseNet121)
    print("Extracting features using pre-trained VGG16...")
    train_image_features = extract_vgg_features(train_images)
    val_image_features = extract_vgg_features(val_images)
    test_image_features = extract_vgg_features(test_images)
    
    # 6. Apply genetic algorithm for feature selection
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
    train_features_selected = train_image_features[:, selected_features.astype(bool)]
    val_features_selected = val_image_features[:, selected_features.astype(bool)]
    test_features_selected = test_image_features[:, selected_features.astype(bool)]
    
    # 7. Calculate class weights for handling imbalance
    class_weights = calc_class_weights(train_labels)
    
    # 8. Create and train model
    print("Creating and training model...")
    model = create_improved_model(
        input_shape=train_features_selected.shape[1],
        num_classes=train_labels.shape[1]
    )
    
    # Define callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ]
    
    # Create focal loss with class weights for imbalanced data
    focal_loss = BinaryFocalCrossentropy(
        gamma=2.0,
        apply_class_balancing=True,  # Automatically apply class balancing
        from_logits=False
    )
    
    # Update the model's loss function with class weights
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=focal_loss,
        metrics=['accuracy', 
                AUC(multi_label=True), 
                F1Score(threshold=0.5, average='weighted'),
                Precision(), 
                Recall()]
    )
    
    # Train with class weights
    history = model.fit(
        train_features_selected, 
        train_labels,
        validation_data=(val_features_selected, val_labels),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # 9. Plot training history
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
    plt.savefig('VGG+GA_training_history.png')
    plt.show()
    
    # 10. Optimize thresholds per class
    print("Optimizing classification thresholds...")
    optimal_thresholds = optimize_thresholds(model, val_features_selected, val_labels)
    
    # 11. Final evaluation
    print("Evaluating model on test set...")
    test_results = model.evaluate(test_features_selected, test_labels)
    print(f"Test results: {dict(zip(model.metrics_names, test_results))}")
    
    # Detailed evaluation with optimized thresholds
    pred_binary, pred_proba = evaluate_model(
        model, test_features_selected, test_labels, label_names, 
        thresholds=optimal_thresholds
    )

if __name__ == "__main__":
    main()