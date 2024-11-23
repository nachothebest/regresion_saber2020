from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso


from model.config.core import config
from model.processing import features as pp



# Define the full pipeline
saber2020_pipe = Pipeline(
    [
        # Step 1: Impute missing values with "most frequent" for all columns
        ("imputer", SimpleImputer(strategy=config.model_config.strategy)),
        
        # Step 2: Apply OneHotEncoder to all columns
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        
        # Step 3: Lasso Regression
        ("Lasso", Lasso(alpha=config.model_config.alpha)),
    ]
)
