import os
from datetime import datetime
from metrics_tracker import MetricsTracker

# Import pipeline stages
from notice_collection.collect import collect_notices
from comment_collection.collect import collect_comments
from comment_ocr.ocr_engine import run_ocr
from comment_analysis.analyze import analyze_comments
from comment_standardization.standardize import standardize_comments

# Config
EXPERIMENTS_DIR = "data/experiments/runs"

class Orchestrator:
    def __init__(self):
        # Create a timestamped experiment folder
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.run_dir = os.path.join(EXPERIMENTS_DIR, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Initialize metrics tracker
        self.metrics = MetricsTracker(self.run_dir)

    def run_pipeline(self):
        print(f"Starting experiment: {self.run_dir}")
        self.metrics.log_event("experiment_start")

        # -----------------------
        # Stage 1: Notice Collection
        # -----------------------
        self.metrics.log_event("notice_collection_start")
        notices = collect_notices(output_dir=self.run_dir)
        self.metrics.log_event("notice_collection_end", count=len(notices))

        # -----------------------
        # Stage 2: Comment Collection
        # -----------------------
        self.metrics.log_event("comment_collection_start")
        comments = collect_comments(notices, output_dir=self.run_dir)
        self.metrics.log_event("comment_collection_end", count=len(comments))

        # -----------------------
        # Stage 3: Comment OCR
        # -----------------------
        self.metrics.log_event("comment_ocr_start")
        ocr_texts = run_ocr(comments, output_dir=self.run_dir)
        self.metrics.log_event("comment_ocr_end", count=len(ocr_texts))

        # -----------------------
        # Stage 4: Comment Analysis
        # -----------------------
        self.metrics.log_event("comment_analysis_start")
        analyzed_comments = analyze_comments(ocr_texts, output_dir=self.run_dir)
        self.metrics.log_event("comment_analysis_end", count=len(analyzed_comments))

        # -----------------------
        # Stage 5: Comment Standardization
        # -----------------------
        self.metrics.log_event("comment_standardization_start")
        standardized_comments = standardize_comments(analyzed_comments, output_dir=self.run_dir)
        self.metrics.log_event("comment_standardization_end", count=len(standardized_comments))

        # -----------------------
        # Save final dataset
        # -----------------------
        self.metrics.save_metrics()
        print(f"Pipeline finished. Experiment data saved in {self.run_dir}")


if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_pipeline()