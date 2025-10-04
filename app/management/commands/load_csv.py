import csv
import os

from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import transaction

from app.models import JobPosting, FraudPrediction

class Command(BaseCommand):
    """
    Usage
    -----
    python manage.py load_csv                             # loads data/fake_job_postings.csv
    python manage.py load_csv --file data/myfile.csv      # load another CSV
    python manage.py load_csv --clear                     # wipe tables then load
    """

    help = "Load job-posting data from a CSV file into the database"

    def add_arguments(self, parser):
        parser.add_argument(
            "--file",
            type=str,
            default="data/fake_job_postings.csv",
            help="Path to CSV file (default: data/fake_job_postings.csv)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete existing JobPosting / FraudPrediction rows first",
        )

    def handle(self, *args, **options):
        csv_file = options["file"]
        csv_path = os.path.join(settings.BASE_DIR, csv_file)

        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f"CSV file not found: {csv_path}"))
            return

        if options["clear"]:
            self.stdout.write("Clearing existing data …")
            FraudPrediction.objects.all().delete()
            JobPosting.objects.all().delete()
            self.stdout.write(self.style.SUCCESS("Existing data deleted."))

        self.stdout.write(f"Loading data from {csv_path} …")

        job_list = []
        fraud_list = []
        batch_size = 500  # Adjust based on your system's memory
        loaded = 0
        errors = 0

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)

            for row_num, row in enumerate(reader, start=1):
                try:
                    # Boolean conversions
                    telecommuting = self._as_bool(row.get("telecommuting"))
                    has_logo = self._as_bool(row.get("has_company_logo"))
                    has_questions = self._as_bool(row.get("has_questions"))
                    fraudulent = self._as_bool(row.get("fraudulent"))

                    # Prepare JobPosting instance
                    job = JobPosting(
                        title=row.get("title", "").strip(),
                        location=row.get("location", "").strip(),
                        department=row.get("department", "").strip(),
                        salary_range=row.get("salary_range", "").strip(),
                        company_profile=row.get("company_profile", "").strip(),
                        description=row.get("description", "").strip(),
                        requirements=row.get("requirements", "").strip(),
                        benefits=row.get("benefits", "").strip(),
                        telecommuting=telecommuting,
                        has_company_logo=has_logo,
                        has_questions=has_questions,
                        employment_type=row.get("employment_type", "").strip(),
                        required_experience=row.get("required_experience", "").strip(),
                        required_education=row.get("required_education", "").strip(),
                        industry=row.get("industry", "").strip(),
                        function=row.get("function", "").strip(),
                        data_source="CSV",
                    )

                    job_list.append(job)
                    fraud_list.append(fraudulent)  # Temporarily store for later association

                    if len(job_list) >= batch_size:
                        self._process_batch(job_list, fraud_list)
                        loaded += len(job_list)
                        self.stdout.write(f"Loaded {loaded} rows …")
                        job_list.clear()
                        fraud_list.clear()

                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    self.stdout.write(self.style.WARNING(f"Row {row_num}: {exc} - Skipping row."))

            # Process any remaining rows
            if job_list:
                self._process_batch(job_list, fraud_list)
                loaded += len(job_list)

        self.stdout.write(self.style.SUCCESS(f"Finished: {loaded} rows imported, {errors} errors."))

    def _process_batch(self, job_list, fraud_list):
        with transaction.atomic():
            created_jobs = JobPosting.objects.bulk_create(job_list)
            prediction_objs = [
                FraudPrediction(
                    job_posting=created_jobs[i],
                    is_fraudulent=fraudulent,
                    confidence_score=1.0,
                    fraud_probability=1.0 if fraudulent else 0.0,
                    sentiment_score=0.0,
                    risk_level="High" if fraudulent else "Low",
                )
                for i, fraudulent in enumerate(fraud_list)
            ]
            FraudPrediction.objects.bulk_create(prediction_objs)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        value = str(value).strip().lower()
        return value in {"1", "true", "yes", "on"}
