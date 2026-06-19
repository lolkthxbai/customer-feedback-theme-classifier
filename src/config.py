"""Shared configuration constants for the complaint classifier."""

BASE_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

TEXT_COLUMN = "Consumer complaint narrative"
DEFAULT_TARGET_COLUMN = "Product"
OPTIONAL_TARGET_COLUMNS = ["Product", "Issue"]

MODEL_PATH = "models/complaint_theme_model.joblib"

MIN_TEXT_LENGTH = 20
MAX_ROWS_FOR_TRAINING = 50000
TEST_SIZE = 0.2
RANDOM_STATE = 42

TFIDF_MAX_FEATURES = 10000
TFIDF_STOP_WORDS = "english"
TFIDF_NGRAM_RANGE = (1, 2)
