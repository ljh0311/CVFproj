# Plantastic: Plant Disease Classification

## Background
Plant diseases pose a significant threat to agricultural productivity and food security. Early detection and accurate diagnosis of plant diseases are crucial for effective disease management. Traditional methods of disease identification rely heavily on human expertise and can be time-consuming and subjective. This project leverages deep learning and computer vision techniques to automate the detection and classification of plant diseases from leaf images.

Key challenges addressed:
- Manual disease identification is time-consuming and requires expert knowledge
- Early disease detection is critical for preventing crop losses
- Need for accessible and accurate diagnostic tools for farmers

## Research Objectives
Our project aims to:
• Develop a deep learning system for accurate plant disease classification
• Design intuitive interfaces for easy disease detection
• Implement and compare multiple state-of-the-art model architectures
• Create a collaborative platform for dataset expansion
• Deliver real-time disease detection with confidence metrics

## Methodology/Proposed Approach

### Dataset
- Primary dataset: New Plant Diseases Dataset (87,000 images, 38 classes)
- Secondary dataset: Landscape Pictures Dataset (for robustness testing)
- Data augmentation techniques:
  - Random crops, flips, and rotations
  - Color jittering and perspective transforms
  - Random erasing for improved generalization

### Model Architecture
1. **ResNet-50**
   - Pre-trained on ImageNet
   - Modified final layer for 38-class classification
   - Fine-tuned on plant disease dataset

2. **EfficientNet-B0**
   - Lightweight architecture for mobile deployment
   - Transfer learning with frozen early layers
   - Custom classifier with dropout for regularization

### Implementation
1. **Training Pipeline**
   - Batch size optimization for memory efficiency
   - Learning rate scheduling with ReduceLROnPlateau
   - Checkpoint saving for best model preservation
   - Cross-entropy loss optimization

2. **User Interfaces**
   - Web interface (Flask)
   - Desktop GUI (Tkinter)
   - Real-time prediction visualization
   - Dataset management tools

## Results and Analysis

### Model Performance
- ResNet-50:
  - Training accuracy: 98.5%
  - Validation accuracy: 96.2%
  - Test accuracy: 95.8%

- EfficientNet-B0:
  - Training accuracy: 97.8%
  - Validation accuracy: 95.9%
  - Test accuracy: 95.1%

### Key Features
1. **Disease Detection**
   - Real-time classification
   - Confidence score indication
   - Support for multiple image formats

2. **User Experience**
   - Intuitive web interface
   - Comprehensive desktop GUI
   - Batch processing capability
   - Community contribution system

## Conclusions and Reflection

### Achievements
1. Successfully implemented dual-model architecture for plant disease classification
2. Created user-friendly interfaces for both web and desktop environments
3. Achieved high accuracy while maintaining practical usability
4. Developed robust dataset management and contribution system

### Future Improvements
1. Expand dataset to include more plant species and disease types
2. Implement model ensemble methods for improved accuracy
3. Add mobile application support
4. Integrate with IoT devices for automated monitoring

### Impact
The project provides an accessible tool for farmers and agricultural professionals to quickly identify plant diseases, enabling early intervention and improved crop management. The community contribution feature ensures continuous improvement of the system's capabilities.

## References
1. New Plant Diseases Dataset (Kaggle)
2. Landscape Pictures Dataset (Kaggle)
3. ResNet: Deep Residual Learning for Image Recognition
4. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks