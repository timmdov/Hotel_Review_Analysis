import os
from src.preprocessing.run_text_preprocessing_pipeline import run_text_preprocessing_pipeline
from src.topic_modeling.run_bertopic_modeling import run_bertopic_modeling
from src.utils.config.paths import TOPIC_MODELING_DIR, STEP8_MODEL_READY
from src.utils.io.logger import get_logger

logger = get_logger(__name__)

def run_preprocessing_and_modeling():
    logger.info("Starting Full Pipeline: Preprocessing + BERTopic")

    # Step 1–8: Preprocessing pipeline
    run_text_preprocessing_pipeline()

    # BERTopic modeling
    output_path = os.path.join(TOPIC_MODELING_DIR, "bertopic_results.csv")
    run_bertopic_modeling(input_path=STEP8_MODEL_READY, output_path=output_path)

    logger.info("run_preprocessing_and_modeling pipeline completed successfully.")

if __name__ == "__main__":
    run_preprocessing_and_modeling()