# World Bank Economic Analysis – Indonesia GDP Forecasting

This project explores global, regional, and Indonesia-level economic data from the **World Bank dataset**, 
with a focus on forecasting **Indonesia’s GDP per capita** using both **SARIMAX** (time series model) 
and **XGBoost** (machine learning model).

The dataset used in this project comes from Kaggle:  
👉 [Global Economic Indicators (2010–2025)](https://www.kaggle.com/datasets/tanishksharma9905/global-economic-indicators-20102025) by *Tanishk Sharma*.

## Forecasting Results
The following visualizations highlight Indonesia’s GDP per capita forecasting results:

### SARIMAX Model
![SARIMAX Forecast](results/indonesia%20GDP%20per%20capita%20(current%20USD)%20SARIMAX.png)

### XGBoost Model
![XGBoost Forecast](results/indonesia%20GDP%20per%20capita%20(Actual%20vs%20XGB%20Predicted).png)

## Key Takeaways
- **SARIMAX** captures the overall trend well, though R² is moderate.  
- **XGBoost** achieves a higher fit (R² ≈ 0.99) by leveraging lag features and exogenous variables.  
- Together, they provide complementary perspectives on Indonesia’s GDP trajectory.  

---

💡 This repo is structured for both **exploration (via notebooks)** and **reproducibility (via scripts)**.
