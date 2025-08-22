import csv
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from app.models import JobPosting, FraudPrediction

class Command(BaseCommand):
    help = 'Load job posting data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            default='data/fake_job_postings.csv',
            help='Path to CSV file (default: data/fake_job_postings.csv)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before loading'
        )

    def handle(self, *args, **options):
        csv_file = options['file']
        csv_path = os.path.join(settings.BASE_DIR, csv_file)
        
        if not os.path.exists(csv_path):
            self.stdout.write(
                self.style.ERROR(f'CSV file not found: {csv_path}')
            )
            return

        if options['clear']:
            self.stdout.write('Clearing existing data...')
            FraudPrediction.objects.all().delete()
            JobPosting.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Data cleared.'))

        self.stdout.write(f'Loading data from {csv_path}...')
        
        loaded_count = 0
        error_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=1):
                try:
                    # Convert string boolean values
                    telecommuting = self.str_to_bool(row.get('telecommuting', '0'))
                    has_company_logo = self.str_to_bool(row.get('has_company_logo', '0'))
                    has_questions = self.str_to_bool(row.get('has_questions', '0'))
                    fraudulent = self.str_to_bool(row.get('fraudulent', '0'))
                    
                    # Create job posting
                    job_posting = JobPosting.objects.create(
                        title=row.get('title', '').strip(),
                        location=row.get('location', '').strip(),
                        department=row.get('department', '').strip(),
                        salary_range=row.get('salary_range', '').strip(),
                        company_profile=row.get('company_profile', '').strip(),
                        description=row.get('description', '').strip(),
                        requirements=row.get('requirements', '').strip(),
                        benefits=row.get('benefits', '').strip(),
                        telecommuting=telecommuting,
                        has_company_logo=has_company_logo,
                        has_questions=has_questions,
                        employment_type=row.get('employment_type', '').strip(),
                        required_experience=row.get('required_experience', '').strip(),
                        required_education=row.get('required_education', '').strip(),
                        industry=row.get('industry', '').strip(),
                        function=row.get('function', '').strip(),
                    )
                    
                    # Create corresponding fraud prediction with actual label
                    FraudPrediction.objects.create(
                        job_posting=job_posting,
                        is_fraudulent=fraudulent,
                        confidence_score=1.0,  # This is ground truth
                        fraud_probability=1.0 if fraudulent else 0.0,
                        sentiment_score=0.0,  # Will be calculated by ML service
                        risk_level='High' if fraudulent else 'Low'
                    )
                    
                    loaded_count += 1
                    
                    if loaded_count % 100 == 0:
                        self.stdout.write(f'Loaded {loaded_count} records...')
                        
                except Exception as e:
                    error_count += 1
                    self.stdout.write(
                        self.style.WARNING(f'Error on row {row_num}: {str(e)}')
                    )
                    continue

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully loaded {loaded_count} job postings. '
                f'Errors: {error_count}'
            )
        )

    def str_to_bool(self, value):
        """Convert string values to boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        if isinstance(value, (int, float)):
            return bool(value)
        return False
