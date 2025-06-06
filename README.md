<div align="justify">
  
# Korea Population: Korea Demographic Projections

## Introduction:

Population forecasting plays a crucial role in planning and decision-making across various domains, including urban development, resource allocation, and public policy. However, existing methodologies often fail to fully capture the complex spatiotemporal patterns present in historical data. To address this limitation, a novel approach is proposed that leverages advanced architectures capable of parallel processing to simultaneously extract spatial and temporal features. The proposed method integrates multiple feature extraction techniques to capture both spatial and temporal dynamics effectively. These features are combined through a fusion mechanism, creating a comprehensive representation of the data. To mitigate the challenges of redundant information and potential overfitting, an attention-based mechanism is employed for optimal feature refinement. This ensures that only the most relevant features are utilized for population projection. Extensive experiments demonstrate the effectiveness of this approach, highlighting its ability to learn discriminative features and improve forecasting accuracy.

## Assoicated Articles of the Project:
### Article 1:
The research introduces a Dual-Stream Network (DSN) for population forecasting that integrates Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) to simultaneously extract spatiotemporal features from historical data. These features are fused into a comprehensive vector, which is then refined using a Self-Attention Mechanism (SAM) to reduce redundancy and prevent overfitting. Experimental results show that this approach enhances the model's ability to capture discriminative patterns and improve the accuracy of population projections

![](Materials/Framework1.svg)


### Article 2:
A dual-stream network for population projection is proposed, integrating 1D Convolutional Neural Networks (CNN), Bidirectional Long Short-Term Memory (BiLSTM), and an Additive Attention Mechanism (Dual CSA). The Dual CSA extracts spatiotemporal features in parallel from historical data, with the 1D-CNN and BiLSTM processing the data simultaneously. The attention mechanism is applied to select important features from the BiLSTM output, which are then concatenated with the CNN output to form a comprehensive feature vector. Extensive experiments are conducted to identify the optimal model for accurate population forecasting

![](Materials/Framework.svg)

## Statistical Methods:

### Decomposition Methods:
Decomposition techniques, including additive and STL methods, are employed to analyze population data by separating it into four distinct components: observed, trend, seasonal, and residual. The observed component represents the raw recorded data, while the trend reveals long-term patterns, highlighting overall changes over time. The seasonal component uncovers recurring cycles or periodic variations, and the residual isolates irregular or random fluctuations, helping to detect anomalies and refine analysis. This approach provides a detailed understanding of the data's structure, enabling more accurate insights and projections.

- **Additive Decomposition**
![](Materials/KoreMonthly_additive.PNG)

- **STL Decomosition**
![](STL_Decomposition/STL_Decomposition.png)


### Feature Importance
In the context of population forecasting is used to identify which factors or variables most significantly influence population trends and projections. By analyzing the contribution of different features, such as demographic, economic, or environmental factors, it becomes possible to understand how each element affects population growth or decline. This process helps in selecting the most relevant features for predictive models, improving their accuracy and interpretability. Understanding feature importance in population data can guide policy decisions, resource allocation, and urban planning by highlighting the key drivers of demographic change. Techniques used to determine feature importance include:

- **PCA**
![](PCA/PCA.png)

- **ICA**
![](ICA/ICA.png)


### Features Selection
Feature selection in population forecasting is the process of identifying and selecting the most relevant features from a larger set of data, which contribute significantly to predicting population trends and projections. By eliminating irrelevant, redundant, or highly correlated features, feature selection helps improve model efficiency, reduce overfitting, and enhance the interpretability of the model. It also speeds up computation by reducing the dimensionality of the dataset. Common techniques for feature selection include filter methods, wrapper methods, and embedded methods, which evaluate features based on statistical tests, model performance, or feature importance scores. Effective feature selection is crucial for building robust and accurate models that can provide reliable insights for policy and planning decisions.

- **ANN**
![](ANN_Based_Feature_Selection/ANN.png)

</div>

