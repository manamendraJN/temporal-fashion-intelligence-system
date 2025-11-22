import React, { useState, useEffect } from 'react';
import { Upload, Activity, Ruler, ShoppingBag, User, Weight, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { apiService } from './services/api';
import ImageUpload from './components/ImageUpload';
import MeasurementsDisplay from './components/MeasurementsDisplay';
import HealthInsights from './components/HealthInsights';
import SizeRecommendations from './components/SizeRecommendations';
import MaskPreview from './components/MaskPreview';
import './App.css';

function App() {
  // State management
  const [frontImage, setFrontImage] = useState(null);
  const [sideImage, setSideImage] = useState(null);
  const [frontPreview, setFrontPreview] = useState(null);
  const [sidePreview, setSidePreview] = useState(null);
  const [weight, setWeight] = useState('');
  const [gender, setGender] = useState('male');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [showFrontPreview, setShowFrontPreview] = useState(false);
  const [showSidePreview, setShowSidePreview] = useState(false);
  const [frontMaskData, setFrontMaskData] = useState(null);
  const [sideMaskData, setSideMaskData] = useState(null);
  const [processingPreview, setProcessingPreview] = useState(false);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const health = await apiService.healthCheck();
      if (health.status === 'healthy' && health.model_loaded) {
        setApiStatus('connected');
      } else {
        setApiStatus('error');
        setError('Backend API is not ready. Please start the backend server.');
      }
    } catch (err) {
      setApiStatus('error');
      setError('Cannot connect to backend API. Make sure it is running on http://localhost:5000');
    }
  };

  // Handle image uploads
const handleImageUpload = async (e, type) => {
  const file = e.target.files[0];
  if (file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError('Image size must be less than 10MB');
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    
    if (type === 'front') {
      setFrontImage(file);
      setFrontPreview(previewUrl);
      
      // Generate preview
      setProcessingPreview(true);
      try {
        const maskData = await apiService.previewMask(file);
        if (maskData.success) {
          setFrontMaskData(maskData.data);
          setShowFrontPreview(true);
        }
      } catch (err) {
        console.error('Preview generation failed:', err);
        setError('Failed to generate preview. You can still proceed with analysis.');
      } finally {
        setProcessingPreview(false);
      }
    } else {
      setSideImage(file);
      setSidePreview(previewUrl);
      
      // Generate preview
      setProcessingPreview(true);
      try {
        const maskData = await apiService.previewMask(file);
        if (maskData.success) {
          setSideMaskData(maskData.data);
          setShowSidePreview(true);
        }
      } catch (err) {
        console.error('Preview generation failed:', err);
        setError('Failed to generate preview. You can still proceed with analysis.');
      } finally {
        setProcessingPreview(false);
      }
    }
    
    setError(null);
  }
};

  // Remove image
  const handleRemoveImage = (type) => {
    if (type === 'front') {
      setFrontImage(null);
      setFrontPreview(null);
    } else {
      setSideImage(null);
      setSidePreview(null);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validation
    if (!frontImage || !sideImage) {
      setError('Please upload both front and side images');
      return;
    }

    if (weight && (isNaN(weight) || parseFloat(weight) <= 0)) {
      setError('Please enter a valid weight');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Call complete analysis API
      const result = await apiService.completeAnalysis(
        frontImage,
        sideImage,
        weight ? parseFloat(weight) : null,
        gender
      );

      if (result.success) {
        setResults(result.data);
        
        // Scroll to results
        setTimeout(() => {
          document.getElementById('results-section')?.scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
          });
        }, 100);
      } else {
        setError(result.error || 'Analysis failed. Please try again.');
      }
    } catch (err) {
      console.error('Analysis error:', err);
      setError(
        err.response?.data?.error || 
        err.message || 
        'Failed to analyze images. Please check your backend server.'
      );
    } finally {
      setLoading(false);
    }
  };

  // Reset form
  const handleReset = () => {
    setFrontImage(null);
    setSideImage(null);
    setFrontPreview(null);
    setSidePreview(null);
    setWeight('');
    setGender('male');
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100">
      {/* Header */}
      <header className="bg-white shadow-lg border-b-4 border-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="bg-primary-600 p-3 rounded-xl shadow-md">
                <Ruler className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">
                  Body Measurement AI
                </h1>
                <p className="text-gray-600 mt-1">
                  AI-powered body measurement from silhouette images
                </p>
              </div>
            </div>
            
            {/* API Status Indicator */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-gray-100">
              {apiStatus === 'connected' && (
                <>
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span className="text-sm font-medium text-green-700">Connected</span>
                </>
              )}
              {apiStatus === 'checking' && (
                <>
                  <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                  <span className="text-sm font-medium text-blue-700">Checking...</span>
                </>
              )}
              {apiStatus === 'error' && (
                <>
                  <XCircle className="w-5 h-5 text-red-600" />
                  <span className="text-sm font-medium text-red-700">Disconnected</span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload Section */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-200">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
              <Upload className="w-7 h-7 text-primary-600" />
              Upload Body Silhouette Images
            </h2>
            {(frontImage || sideImage) && (
              <button
                type="button"
                onClick={handleReset}
                className="text-sm text-gray-600 hover:text-red-600 font-medium transition"
              >
                Reset All
              </button>
            )}
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Image Uploads */}
            <div className="grid md:grid-cols-2 gap-6">
              <ImageUpload
                label="Front View (Required) *"
                image={frontImage}
                preview={frontPreview}
                onChange={(e) => handleImageUpload(e, 'front')}
                onRemove={() => handleRemoveImage('front')}
              />
              
              <ImageUpload
                label="Side View (Required) *"
                image={sideImage}
                preview={sidePreview}
                onChange={(e) => handleImageUpload(e, 'side')}
                onRemove={() => handleRemoveImage('side')}
              />
            </div>

            {/* Mask Preview Modals */}
            {showFrontPreview && frontMaskData && (
              <MaskPreview
                original={frontPreview}
                preview={frontMaskData.preview}
                mask={frontMaskData.mask}
                onAccept={() => setShowFrontPreview(false)}
                onRetry={() => {
                  setShowFrontPreview(false);
                  handleRemoveImage('front');
                }}
              />
            )}

            {showSidePreview && sideMaskData && (
              <MaskPreview
                original={sidePreview}
                preview={sideMaskData.preview}
                mask={sideMaskData.mask}
                onAccept={() => setShowSidePreview(false)}
                onRetry={() => {
                  setShowSidePreview(false);
                  handleRemoveImage('side');
                }}
              />
            )}

            {processingPreview && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white rounded-xl p-8 text-center">
                  <Loader2 className="w-12 h-12 text-primary-600 animate-spin mx-auto mb-4" />
                  <p className="text-lg font-semibold">Processing image...</p>
                  <p className="text-sm text-gray-600">Removing background and creating mask</p>
                </div>
              </div>
            )}

            {/* Additional Information */}
            <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">
                Additional Information
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                {/* Weight Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <Weight className="w-4 h-4 inline mr-2 text-primary-600" />
                    Weight (kg) - Optional
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    max="300"
                    value={weight}
                    onChange={(e) => setWeight(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent transition"
                    placeholder="e.g., 70.5"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Required for BMI calculation and health insights
                  </p>
                </div>

                {/* Gender Selection */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    <User className="w-4 h-4 inline mr-2 text-primary-600" />
                    Gender
                  </label>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent transition"
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Used for size recommendations
                  </p>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || !frontImage || !sideImage || apiStatus !== 'connected'}
              className="w-full bg-gradient-to-r from-primary-600 to-purple-600 text-white py-4 rounded-xl font-bold text-lg hover:from-primary-700 hover:to-purple-700 transition-all transform hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg flex items-center justify-center gap-3"
            >
              {loading ? (
                <>
                  <Loader2 className="w-6 h-6 animate-spin" />
                  Analyzing Images...
                </>
              ) : (
                <>
                  <Activity className="w-6 h-6" />
                  Analyze Body Measurements
                </>
              )}
            </button>
          </form>

          {/* Error Message */}
          {error && (
            <div className="mt-6 bg-red-50 border-l-4 border-red-500 text-red-700 px-6 py-4 rounded-lg flex items-start gap-3">
              <XCircle className="w-6 h-6 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          )}
        </div>

        {/* Results Section */}
        {results && (
          <div id="results-section" className="space-y-8 animate-fade-in">
            {/* Success Message */}
            <div className="bg-green-50 border-l-4 border-green-500 text-green-700 px-6 py-4 rounded-lg flex items-center gap-3">
              <CheckCircle className="w-6 h-6" />
              <div>
                <p className="font-semibold">Analysis Complete!</p>
                <p className="text-sm">Your body measurements have been successfully analyzed.</p>
              </div>
            </div>

            {/* Measurements */}
            {results.measurements && (
              <MeasurementsDisplay measurements={results.measurements} />
            )}

            {/* Health Insights */}
            {results.health && (
              <HealthInsights healthData={results.health} />
            )}

            {/* Size Recommendations */}
            {results.size_recommendations && (
              <SizeRecommendations sizeData={results.size_recommendations} />
            )}

            {/* Model Info */}
            {results.model && (
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
                <p className="text-sm text-gray-600">
                  <span className="font-semibold">Model Used:</span> {results.model}
                </p>
              </div>
            )}
          </div>
        )}

        {/* Instructions */}
        {!results && (
        <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-lg">
          <h4 className="font-bold text-blue-900 mb-3">📸 Photo Guidelines for Best Results</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800">
            <div>
              <p className="font-semibold mb-2">✅ DO:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Use plain white or solid color background</li>
                <li>Stand straight with arms slightly away from body</li>
                <li>Wear fitted clothing</li>
                <li>Ensure good, even lighting</li>
                <li>Take front and side views from same distance</li>
              </ul>
            </div>
            <div>
              <p className="font-semibold mb-2">❌ AVOID:</p>
              <ul className="list-disc list-inside space-y-1">
                <li>Busy or cluttered backgrounds</li>
                <li>Arms crossed or touching body</li>
                <li>Baggy or loose clothing</li>
                <li>Poor lighting or shadows</li>
                <li>Partial body shots</li>
              </ul>
            </div>
          </div>
        </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white mt-16 py-8 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-gray-600">
            <span className="font-semibold">Body Measurement AI</span> v1.0.0
          </p>
          <p className="text-gray-500 text-sm mt-2">
            Built with ❤️ by <span className="font-semibold">manamendraJN</span> | November 2025
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;