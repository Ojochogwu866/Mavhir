# Model Information

## Model Development Methodology

### Data Collection
- **Ames Data**: Curated from published literature and databases
- **Carcinogenicity Data**: NTP and CPDB databases
- **Quality Control**: Manual curation and expert review

### Feature Engineering
- **Descriptors**: 200+ Mordred molecular descriptors
- **Selection**: Variance filtering and correlation analysis
- **Scaling**: StandardScaler normalization

### Model Training
- **Algorithm Selection**: Cross-validation comparison
- **Hyperparameter Tuning**: Grid search with 5-fold CV
- **Validation**: External test set evaluation

### Performance Metrics
- **Accuracy**: Balanced accuracy for imbalanced datasets
- **AUC-ROC**: Area under receiver operating characteristic
- **Precision/Recall**: For regulatory decision making
- **Confidence**: Based on prediction probability

## Model Limitations

### Applicability Domain
- Organic compounds with MW 50-1000 Da
- Standard drug-like chemical space
- May not be suitable for:
  - Metal complexes
  - Large biologics
  - Novel chemical classes

### Data