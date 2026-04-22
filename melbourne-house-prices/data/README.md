# Data

## Melbourne Housing Dataset

This project uses the **Melbourne Housing Snapshot** dataset from Kaggle.

### How to get the data

1. Go to: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
2. Download `melb_data.csv`
3. Place it in this `data/` folder

### Dataset Overview

| Property        | Value              |
|----------------|--------------------|
| Source          | Kaggle             |
| Rows            | 13,580 houses      |
| Columns         | 21 features        |
| Target variable | `Price` (AUD)      |

### Column Descriptions

| Column         | Type    | Description                              |
|----------------|---------|------------------------------------------|
| `Rooms`        | int     | Number of rooms                          |
| `Price`        | int     | Sale price in AUD (**target**)           |
| `Distance`     | float   | Distance from Melbourne CBD (km)         |
| `Bathroom`     | int     | Number of bathrooms                      |
| `Car`          | int     | Number of car spaces                     |
| `Landsize`     | int     | Land size in square metres               |
| `BuildingArea` | float   | Building size in square metres           |
| `YearBuilt`    | float   | Year the property was built              |
