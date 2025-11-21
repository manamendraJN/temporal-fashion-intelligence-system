"""
Body Measurement AI - Flask API
Author: manamendraJN
Date: 2025-11-18
Version: 1.0.0
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from pathlib import Path
from image_processor import image_processor

from config import Config
from model import ModelInference
from health import HealthAnalyzer
from size_recommender import SizeRecommender
from utils import (
    allowed_file, 
    decode_base64_image, 
    format_measurements,
    validate_measurements,
    create_error_response,
    create_success_response
)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

# Initialize services
print("🚀 Initializing Body Measurement AI API...")
print(f"📍 Model directory: {Config.MODEL_DIR}")

# Check if models exist
if not Config.MODEL_DIR.exists():
    print(f"⚠️ Creating model directory: {Config.MODEL_DIR}")
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load default model
try:
    model_inference = ModelInference(
        model_name=Config.DEFAULT_MODEL,
        device='cuda' if os.getenv('USE_GPU', 'False') == 'True' else 'cpu'
    )
    print(f"✅ Default model loaded: {Config.DEFAULT_MODEL}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model_inference = None

# Initialize health analyzer and size recommender
health_analyzer = HealthAnalyzer()
size_recommender = SizeRecommender()

print("✅ API initialized successfully!")

# ==================== ROUTES ====================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'api': Config.API_TITLE,
        'version': Config.API_VERSION,
        'status': 'running',
        'endpoints': {
            'predict': '/predict [POST]',
            'health': '/health [POST]',
            'recommend_size': '/recommend-size [POST]',
            'model_info': '/model-info [GET]',
            'health_check': '/health [GET]'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_inference is not None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if model_inference is None:
        return create_error_response("Model not loaded", 500)
    
    info = model_inference.get_model_info()
    return create_success_response(info, "Model information retrieved")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict body measurements from images
    
    Expected JSON payload:
    {
        "front_image": "base64_encoded_image_or_url",
        "side_image": "base64_encoded_image_or_url",
        "model": "efficientnet-b3" (optional)
    }
    
    Or use multipart/form-data with files:
    - front_image: file
    - side_image: file
    """
    try:
        if model_inference is None:
            return create_error_response("Model not loaded", 500)
        
        # Get images from request
        if request.is_json:
            # JSON request with base64 images
            data = request.get_json()
            
            if 'front_image' not in data or 'side_image' not in data:
                return create_error_response("Missing front_image or side_image", 400)
            
            try:
                front_bytes = decode_base64_image(data['front_image'])
                side_bytes = decode_base64_image(data['side_image'])
            except Exception as e:
                return create_error_response(f"Invalid image format: {str(e)}", 400)
        
        else:
            # Multipart form data with files
            if 'front_image' not in request.files or 'side_image' not in request.files:
                return create_error_response("Missing front_image or side_image files", 400)
            
            front_file = request.files['front_image']
            side_file = request.files['side_image']
            
            if not allowed_file(front_file.filename) or not allowed_file(side_file.filename):
                return create_error_response("Invalid file type. Allowed: png, jpg, jpeg", 400)
            
            front_bytes_raw = front_file.read()
            side_bytes_raw = side_file.read()

            # Process images to create masks
            print("🔄 Processing images to create body masks...")
            front_bytes = image_processor.process_image(front_bytes_raw, Config.IMG_SIZE)
            side_bytes = image_processor.process_image(side_bytes_raw, Config.IMG_SIZE)
            print("✅ Masks created successfully")
        
        # Run inference
        measurements = model_inference.predict(front_bytes, side_bytes)
        
        # Validate measurements
        warnings = validate_measurements(measurements)
        
        # Format response
        formatted_measurements = format_measurements(measurements)
        
        response_data = {
            'measurements': formatted_measurements,
            'model': model_inference.model_config['name'],
            'warnings': warnings if warnings else None
        }
        
        return create_success_response(response_data, "Measurements predicted successfully")
    
    except Exception as e:
        print(f"❌ Error in /predict: {traceback.format_exc()}")
        return create_error_response(f"Prediction error: {str(e)}", 500)

@app.route('/health-insights', methods=['POST'])
def health_insights():
    """
    Get health insights from measurements
    
    Expected JSON payload:
    {
        "measurements": { ... },
        "weight_kg": 70.5,
        "gender": "male" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if 'measurements' not in data:
            return create_error_response("Missing measurements", 400)
        
        measurements = data['measurements']
        
        # Extract values from formatted measurements if needed
        if isinstance(list(measurements.values())[0], dict):
            measurements = {k: v['value'] for k, v in measurements.items()}
        
        weight_kg = data.get('weight_kg')
        gender = data.get('gender', 'male')
        
        # Generate health report
        health_report = health_analyzer.get_full_health_report(
            measurements, 
            weight_kg, 
            gender
        )
        
        return create_success_response(health_report, "Health insights generated")
    
    except Exception as e:
        print(f"❌ Error in /health-insights: {traceback.format_exc()}")
        return create_error_response(f"Health analysis error: {str(e)}", 500)

@app.route('/preview-mask', methods=['POST'])
def preview_mask():
    """
    Preview the mask that will be generated from uploaded image
    
    Expected: multipart/form-data with 'image' file
    Returns: Base64 encoded preview image
    """
    try:
        if 'image' not in request.files:
            return create_error_response("Missing image file", 400)
        
        image_file = request.files['image']
        
        if not allowed_file(image_file.filename):
            return create_error_response("Invalid file type", 400)
        
        image_bytes = image_file.read()
        
        # Process and get preview
        result = image_processor.process_and_preview(image_bytes, Config.IMG_SIZE)
        
        # Convert to base64 for frontend display
        import base64
        preview_base64 = base64.b64encode(result['preview_bytes']).decode('utf-8')
        mask_base64 = base64.b64encode(result['mask_bytes']).decode('utf-8')
        
        return create_success_response({
            'preview': f"data:image/png;base64,{preview_base64}",
            'mask': f"data:image/png;base64,{mask_base64}"
        }, "Preview generated")
    
    except Exception as e:
        print(f"❌ Error in /preview-mask: {traceback.format_exc()}")
        return create_error_response(f"Preview error: {str(e)}", 500)

@app.route('/recommend-size', methods=['POST'])
def recommend_size():
    """
    Recommend clothing size from measurements
    
    Expected JSON payload:
    {
        "measurements": { ... },
        "gender": "male",
        "region": "us" (optional),
        "garment_type": "top" (optional: top, bottom, dress)
    }
    """
    try:
        data = request.get_json()
        
        if 'measurements' not in data:
            return create_error_response("Missing measurements", 400)
        
        measurements = data['measurements']
        
        # Extract values from formatted measurements if needed
        if isinstance(list(measurements.values())[0], dict):
            measurements = {k: v['value'] for k, v in measurements.items()}
        
        gender = data.get('gender', 'male')
        region = data.get('region', 'us')
        garment_type = data.get('garment_type', 'top')
        
        # Get size recommendation
        size_rec = size_recommender.recommend_size(
            measurements, 
            gender, 
            region, 
            garment_type
        )
        
        # Get all size recommendations
        all_sizes = size_recommender.get_all_sizes(measurements, gender, region)
        
        response_data = {
            'primary_recommendation': size_rec,
            'all_recommendations': all_sizes
        }
        
        return create_success_response(response_data, "Size recommendation generated")
    
    except Exception as e:
        print(f"❌ Error in /recommend-size: {traceback.format_exc()}")
        return create_error_response(f"Size recommendation error: {str(e)}", 500)

@app.route('/complete-analysis', methods=['POST'])
@app.route('/complete-analysis', methods=['POST'])
def complete_analysis():
    """
    Complete analysis: measurements + health + size recommendations
    
    Expected payload (multipart or JSON):
    - front_image
    - side_image
    - weight_kg (optional)
    - gender (optional)
    """
    try:
        if model_inference is None:
            return create_error_response("Model not loaded", 500)
        
        # Get images from request
        if 'front_image' not in request.files or 'side_image' not in request.files:
            return create_error_response("Missing front_image or side_image files", 400)
        
        front_file = request.files['front_image']
        side_file = request.files['side_image']
        
        if not allowed_file(front_file.filename) or not allowed_file(side_file.filename):
            return create_error_response("Invalid file type. Allowed: png, jpg, jpeg", 400)
        
        front_bytes = front_file.read()
        side_bytes = side_file.read()
        
        # Get additional parameters from form data
        weight_kg = request.form.get('weight_kg')
        gender = request.form.get('gender', 'male')
        
        # Run inference directly (don't call predict() endpoint)
        measurements = model_inference.predict(front_bytes, side_bytes)
        
        # Validate measurements
        warnings = validate_measurements(measurements)
        
        # Format measurements
        formatted_measurements = format_measurements(measurements)
        
        # Extract measurement values for health analysis
        measurement_values = {k: v['value'] for k, v in formatted_measurements.items()}
        
        # Get health insights (only if weight is provided)
        health_report = None
        if weight_kg:
            try:
                health_report = health_analyzer.get_full_health_report(
                    measurement_values,
                    float(weight_kg),
                    gender
                )
            except Exception as e:
                print(f"⚠️ Health analysis warning: {e}")
        
        # Get size recommendations
        all_sizes = None
        try:
            all_sizes = size_recommender.get_all_sizes(measurement_values, gender, 'us')
        except Exception as e:
            print(f"⚠️ Size recommendation warning: {e}")
        
        # Combine all results
        complete_data = {
            'measurements': formatted_measurements,
            'health': health_report,
            'size_recommendations': all_sizes,
            'model': model_inference.model_config['name'],
            'warnings': warnings if warnings else None
        }
        
        return create_success_response(complete_data, "Complete analysis generated")
    
    except Exception as e:
        print(f"❌ Error in /complete-analysis: {traceback.format_exc()}")
        return create_error_response(f"Analysis error: {str(e)}", 500)

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return create_error_response("Endpoint not found", 404)

@app.errorhandler(500)
def internal_error(e):
    return create_error_response("Internal server error", 500)

# Run server
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 Starting {Config.API_TITLE} v{Config.API_VERSION}")
    print(f"📍 Host: {Config.HOST}:{Config.PORT}")
    print(f"🔧 Debug: {Config.DEBUG}")
    print(f"{'='*60}\n")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )