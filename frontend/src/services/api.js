import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Health check
  healthCheck: async () => {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Get model info
  getModelInfo: async () => {
    try {
      const response = await api.get('/model-info');
      return response.data;
    } catch (error) {
      console.error('Failed to get model info:', error);
      throw error;
    }
  },

  // Predict measurements from images
  predictMeasurements: async (frontImage, sideImage) => {
    try {
      const formData = new FormData();
      formData.append('front_image', frontImage);
      formData.append('side_image', sideImage);

      const response = await api.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Prediction failed:', error);
      throw error;
    }
  },

  // Get health insights
  getHealthInsights: async (measurements, weightKg, gender) => {
    try {
      const response = await api.post('/health-insights', {
        measurements,
        weight_kg: weightKg,
        gender,
      });
      return response.data;
    } catch (error) {
      console.error('Health insights failed:', error);
      throw error;
    }
  },

  // Get size recommendations
  getSizeRecommendations: async (measurements, gender, garmentType = 'top') => {
    try {
      const response = await api.post('/recommend-size', {
        measurements,
        gender,
        garment_type: garmentType,
      });
      return response.data;
    } catch (error) {
      console.error('Size recommendation failed:', error);
      throw error;
    }
  },

  // Preview mask before analysis
  previewMask: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await api.post('/preview-mask', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Preview failed:', error);
      throw error;
    }
  },

  // Complete analysis (all-in-one)
  completeAnalysis: async (frontImage, sideImage, weightKg, gender) => {
    try {
      const formData = new FormData();
      formData.append('front_image', frontImage);
      formData.append('side_image', sideImage);
      if (weightKg) formData.append('weight_kg', weightKg);
      formData.append('gender', gender);

      const response = await api.post('/complete-analysis', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Complete analysis failed:', error);
      throw error;
    }
  },
};

export default api;