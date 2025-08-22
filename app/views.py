from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Count, Q
import json
from datetime import datetime, timedelta
from .models import JobPosting, FraudPrediction
from .ml_service import FraudDetectionService

# Initialize ML service
fraud_detector = FraudDetectionService()

def home(request):
    """Main fraud detection page"""
    return render(request, 'fraud_detection/home.html')

@csrf_exempt
def predict_fraud(request):
    """Handle fraud prediction requests"""
    if request.method == 'POST':
        try:
            if request.content_type == 'application/json':
                data = json.loads(request.body)
            else:
                data = request.POST.dict()
                # Convert checkbox values
                data['telecommuting'] = 'telecommuting' in request.POST
                data['has_company_logo'] = 'has_company_logo' in request.POST
                data['has_questions'] = 'has_questions' in request.POST
            
            # Create job posting
            job_posting = JobPosting.objects.create(
                title=data.get('title', ''),
                location=data.get('location', ''),
                department=data.get('department', ''),
                salary_range=data.get('salary_range', ''),
                company_profile=data.get('company_profile', ''),
                description=data.get('description', ''),
                requirements=data.get('requirements', ''),
                benefits=data.get('benefits', ''),
                telecommuting=data.get('telecommuting', False),
                has_company_logo=data.get('has_company_logo', False),
                has_questions=data.get('has_questions', False),
                employment_type=data.get('employment_type', ''),
                required_experience=data.get('required_experience', ''),
                required_education=data.get('required_education', ''),
                industry=data.get('industry', ''),
                function=data.get('function', '')
            )
            
            # Get prediction
            prediction = fraud_detector.predict_fraud(data)
            
            # Save prediction
            fraud_prediction = FraudPrediction.objects.create(
                job_posting=job_posting,
                is_fraudulent=prediction['is_fraudulent'],
                confidence_score=prediction['confidence'],
                fraud_probability=prediction['fraud_probability'],
                sentiment_score=prediction['sentiment_score'],
                risk_level=prediction['risk_level']
            )
            
            if request.content_type == 'application/json':
                return JsonResponse({
                    'success': True,
                    'prediction': {
                        'is_fraudulent': prediction['is_fraudulent'],
                        'confidence': f"{prediction['confidence']:.1%}",
                        'fraud_probability': f"{prediction['fraud_probability']:.1%}",
                        'sentiment_score': f"{prediction['sentiment_score']:.2f}",
                        'risk_level': prediction['risk_level']
                    },
                    'job_id': job_posting.job_id
                })
            else:
                return render(request, 'fraud_detection/result.html', {
                    'prediction': prediction,
                    'job_posting': job_posting,
                    'fraud_prediction': fraud_prediction
                })
            
        except Exception as e:
            if request.content_type == 'application/json':
                return JsonResponse({'success': False, 'error': str(e)})
            else:
                return render(request, 'fraud_detection/error.html', {'error': str(e)})
    
    return render(request, 'fraud_detection/predict.html')

def analytics_dashboard(request):
    """Analytics dashboard view"""
    # Basic statistics
    total_predictions = FraudPrediction.objects.count()
    fraud_count = FraudPrediction.objects.filter(is_fraudulent=True).count()
    
    # Risk level distribution
    risk_distribution = FraudPrediction.objects.values('risk_level').annotate(
        count=Count('risk_level')
    ).order_by('risk_level')
    
    # Recent predictions with pagination
    recent_predictions = FraudPrediction.objects.select_related('job_posting').order_by('-created_at')
    paginator = Paginator(recent_predictions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Daily fraud detection over last 7 days
    last_week = datetime.now() - timedelta(days=7)
    daily_stats = []
    for i in range(7):
        day = last_week + timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_predictions = FraudPrediction.objects.filter(
            created_at__gte=day_start,
            created_at__lt=day_end
        )
        daily_stats.append({
            'date': day.strftime('%Y-%m-%d'),
            'total': day_predictions.count(),
            'fraud': day_predictions.filter(is_fraudulent=True).count()
        })
    
    # Industry analysis
    industry_stats = JobPosting.objects.exclude(industry='').values('industry').annotate(
        total_jobs=Count('job_id'),
        fraud_count=Count('fraud_prediction', filter=Q(fraud_prediction__is_fraudulent=True))
    ).order_by('-total_jobs')[:10]
    
    context = {
        'total_predictions': total_predictions,
        'fraud_count': fraud_count,
        'legitimate_count': total_predictions - fraud_count,
        'fraud_percentage': (fraud_count / total_predictions * 100) if total_predictions > 0 else 0,
        'risk_distribution': {item['risk_level']: item['count'] for item in risk_distribution},
        'recent_predictions': page_obj,
        'daily_stats': daily_stats,
        'industry_stats': industry_stats
    }
    
    return render(request, 'fraud_detection/dashboard.html', context)

def job_detail(request, job_id):
    """View individual job posting details"""
    try:
        job_posting = JobPosting.objects.get(job_id=job_id)
        fraud_prediction = FraudPrediction.objects.get(job_posting=job_posting)
        
        context = {
            'job_posting': job_posting,
            'fraud_prediction': fraud_prediction
        }
        return render(request, 'fraud_detection/job_detail.html', context)
    except (JobPosting.DoesNotExist, FraudPrediction.DoesNotExist):
        return render(request, 'fraud_detection/error.html', {'error': 'Job not found'})
