from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.utils import timezone
import uuid

class JobPosting(models.Model):
    # Core job posting fields
    job_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=200)
    location = models.CharField(max_length=200, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    salary_range = models.CharField(max_length=100, blank=True, null=True)
    company_profile = models.TextField(blank=True, null=True)
    description = models.TextField()
    requirements = models.TextField(blank=True, null=True)
    benefits = models.TextField(blank=True, null=True)
    telecommuting = models.BooleanField(default=False)
    has_company_logo = models.BooleanField(default=False)
    has_questions = models.BooleanField(default=False)
    employment_type = models.CharField(max_length=50, blank=True, null=True)
    required_experience = models.CharField(max_length=50, blank=True, null=True)
    required_education = models.CharField(max_length=50, blank=True, null=True)
    industry = models.CharField(max_length=100, blank=True, null=True)
    function = models.CharField(max_length=100, blank=True, null=True)
    
    # Enhanced fields for ML learning and tracking
    DATA_SOURCE_CHOICES = [
        ('CSV', 'Original CSV Dataset'),
        ('USER', 'User Submitted'),
        ('API', 'API Submission'),
        ('BULK', 'Bulk Upload'),
        ('SCRAPER', 'Web Scraper')
    ]
    
    data_source = models.CharField(
        max_length=20, 
        choices=DATA_SOURCE_CHOICES, 
        default='USER'
    )
    
    # User interaction tracking
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True, null=True)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Content analysis metadata
    word_count = models.PositiveIntegerField(null=True, blank=True)
    character_count = models.PositiveIntegerField(null=True, blank=True)
    url_count = models.PositiveIntegerField(default=0)
    email_count = models.PositiveIntegerField(default=0)
    phone_count = models.PositiveIntegerField(default=0)
    
    # Quality indicators
    has_company_website = models.BooleanField(default=False)
    has_contact_info = models.BooleanField(default=False)
    has_specific_requirements = models.BooleanField(default=False)
    description_quality_score = models.FloatField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Soft delete
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'job_postings'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['data_source']),
            models.Index(fields=['created_at']),
            models.Index(fields=['industry']),
            models.Index(fields=['employment_type']),
            models.Index(fields=['-created_at', 'data_source']),
        ]

    def __str__(self):
        return f"#{self.job_id} - {self.title}"
    
    def soft_delete(self):
        """Soft delete the job posting"""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save()

class FraudPrediction(models.Model):
    RISK_LEVELS = [
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High'),
    ]
    
    METHOD_CHOICES = [
        ('RULE_BASED', 'Rule-based Algorithm'),
        ('ML_MODEL', 'Machine Learning Model'),
        ('ENSEMBLE', 'Ensemble Method'),
        ('DEEP_LEARNING', 'Deep Learning Model'),
        ('HYBRID', 'Hybrid Approach')
    ]
    
    job_posting = models.OneToOneField(
        JobPosting, 
        on_delete=models.CASCADE,
        related_name='fraud_prediction'
    )
    
    # Core prediction results
    is_fraudulent = models.BooleanField()
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    fraud_probability = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    sentiment_score = models.FloatField(
        validators=[MinValueValidator(-1.0), MaxValueValidator(1.0)]
    )
    risk_level = models.CharField(
        max_length=10, 
        choices=RISK_LEVELS,
        default='Low'
    )
    
    # Enhanced ML tracking fields
    model_version = models.CharField(max_length=50, default='v1.0')
    prediction_method = models.CharField(
        max_length=20,
        choices=METHOD_CHOICES,
        default='RULE_BASED'
    )
    processing_time = models.FloatField(
        null=True, 
        blank=True,
        help_text="Processing time in seconds"
    )
    
    # Feature importance scores (stored as JSON)
    feature_scores = models.JSONField(default=dict, blank=True)
    
    # Model confidence breakdown
    text_confidence = models.FloatField(null=True, blank=True)
    metadata_confidence = models.FloatField(null=True, blank=True)
    sentiment_confidence = models.FloatField(null=True, blank=True)
    
    # Prediction quality metrics
    entropy = models.FloatField(null=True, blank=True)
    prediction_variance = models.FloatField(null=True, blank=True)
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Human verification
    is_verified = models.BooleanField(default=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    verified_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='verified_predictions'
    )

    class Meta:
        db_table = 'fraud_predictions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_fraudulent']),
            models.Index(fields=['risk_level']),
            models.Index(fields=['model_version']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_verified']),
        ]

    def __str__(self):
        return f"Prediction for Job #{self.job_posting.job_id} - {self.risk_level} Risk"

    @property
    def confidence_percentage(self):
        return f"{self.confidence_score * 100:.1f}%"
    
    @property
    def fraud_percentage(self):
        return f"{self.fraud_probability * 100:.1f}%"
    
    def mark_verified(self, user=None):
        """Mark prediction as human-verified"""
        self.is_verified = True
        self.verified_at = timezone.now()
        if user:
            self.verified_by = user
        self.save()

class UserFeedback(models.Model):
    """Store user corrections and feedback to improve model accuracy"""
    
    FEEDBACK_TYPES = [
        ('CORRECTION', 'Fraud Classification Correction'),
        ('QUALITY', 'Prediction Quality Rating'),
        ('FEATURE', 'Feature Importance Feedback'),
        ('GENERAL', 'General Feedback')
    ]
    
    AGREEMENT_LEVELS = [
        ('STRONGLY_AGREE', 'Strongly Agree'),
        ('AGREE', 'Agree'),
        ('NEUTRAL', 'Neutral'),
        ('DISAGREE', 'Disagree'),
        ('STRONGLY_DISAGREE', 'Strongly Disagree')
    ]
    
    prediction = models.ForeignKey(
        FraudPrediction, 
        on_delete=models.CASCADE,
        related_name='user_feedback'
    )
    
    # User correction data
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPES)
    user_correction = models.BooleanField(
        null=True, 
        blank=True,
        help_text="User's correction: True if fraud, False if legitimate"
    )
    
    # Feedback details
    agreement_level = models.CharField(
        max_length=20,
        choices=AGREEMENT_LEVELS,
        null=True,
        blank=True
    )
    confidence_rating = models.IntegerField(
        choices=[(i, f"{i} Star{'s' if i != 1 else ''}") for i in range(1, 6)],
        null=True,
        blank=True,
        help_text="User confidence in their correction (1-5 stars)"
    )
    
    # Detailed feedback
    feedback_reason = models.TextField(
        blank=True,
        help_text="Why the user thinks the prediction is wrong"
    )
    specific_issues = models.JSONField(
        default=list,
        blank=True,
        help_text="Specific issues identified by user"
    )
    
    # User information (anonymous)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    
    # Optional user account (if logged in)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='fraud_feedback'
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Quality control
    is_spam = models.BooleanField(default=False)
    is_helpful = models.BooleanField(null=True, blank=True)

    class Meta:
        db_table = 'user_feedback'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['feedback_type']),
            models.Index(fields=['is_processed']),
            models.Index(fields=['created_at']),
            models.Index(fields=['user_correction']),
        ]

    def __str__(self):
        return f"Feedback for Job #{self.prediction.job_posting.job_id} - {self.feedback_type}"
    
    def mark_processed(self):
        """Mark feedback as processed for model training"""
        self.is_processed = True
        self.processed_at = timezone.now()
        self.save()

class ModelPerformance(models.Model):
    """Track model performance metrics over time"""
    
    model_version = models.CharField(max_length=50)
    model_type = models.CharField(max_length=50, default='fraud_detector')
    
    # Performance metrics
    accuracy = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    precision = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    recall = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    f1_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    auc_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Dataset information
    total_predictions = models.PositiveIntegerField()
    true_positives = models.PositiveIntegerField()
    true_negatives = models.PositiveIntegerField()
    false_positives = models.PositiveIntegerField()
    false_negatives = models.PositiveIntegerField()
    
    # Training information
    training_data_size = models.PositiveIntegerField(null=True, blank=True)
    training_time = models.FloatField(null=True, blank=True)
    feature_count = models.PositiveIntegerField(null=True, blank=True)
    
    # Deployment tracking
    date_evaluated = models.DateTimeField(auto_now_add=True)
    date_deployed = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    
    # Performance notes
    notes = models.TextField(blank=True)
    evaluation_dataset = models.CharField(max_length=100, blank=True)

    class Meta:
        db_table = 'model_performance'
        ordering = ['-date_evaluated']
        indexes = [
            models.Index(fields=['model_version']),
            models.Index(fields=['is_active']),
            models.Index(fields=['date_evaluated']),
        ]

    def __str__(self):
        return f"Model {self.model_version} - Accuracy: {self.accuracy:.3f}"
    
    @property
    def specificity(self):
        """Calculate specificity (true negative rate)"""
        if (self.true_negatives + self.false_positives) == 0:
            return 0
        return self.true_negatives / (self.true_negatives + self.false_positives)

class DataQualityMetrics(models.Model):
    """Track data quality metrics for continuous monitoring"""
    
    METRIC_TYPES = [
        ('COMPLETENESS', 'Data Completeness'),
        ('ACCURACY', 'Data Accuracy'),
        ('CONSISTENCY', 'Data Consistency'),
        ('VALIDITY', 'Data Validity'),
        ('UNIQUENESS', 'Data Uniqueness')
    ]
    
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPES)
    metric_value = models.FloatField()
    threshold_value = models.FloatField()
    is_passing = models.BooleanField()
    
    # Context
    table_name = models.CharField(max_length=50)
    column_name = models.CharField(max_length=50, blank=True)
    date_checked = models.DateTimeField(auto_now_add=True)
    
    # Details
    record_count = models.PositiveIntegerField()
    issue_count = models.PositiveIntegerField(default=0)
    notes = models.TextField(blank=True)

    class Meta:
        db_table = 'data_quality_metrics'
        ordering = ['-date_checked']

    def __str__(self):
        return f"{self.metric_type} for {self.table_name}: {self.metric_value:.3f}"

class PredictionBatch(models.Model):
    """Track batch prediction jobs for bulk processing"""
    
    batch_id = models.UUIDField(default=uuid.uuid4, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Status tracking
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
        ('CANCELLED', 'Cancelled')
    ]
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    
    # Metrics
    total_jobs = models.PositiveIntegerField(default=0)
    processed_jobs = models.PositiveIntegerField(default=0)
    failed_jobs = models.PositiveIntegerField(default=0)
    
    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Configuration
    model_version = models.CharField(max_length=50, default='latest')
    
    # Results summary
    fraud_detected = models.PositiveIntegerField(default=0)
    avg_confidence = models.FloatField(null=True, blank=True)
    
    # Error tracking
    error_message = models.TextField(blank=True)

    class Meta:
        db_table = 'prediction_batches'
        ordering = ['-created_at']

    def __str__(self):
        return f"Batch {self.name} - {self.status}"
    
    @property
    def progress_percentage(self):
        if self.total_jobs == 0:
            return 0
        return (self.processed_jobs / self.total_jobs) * 100
