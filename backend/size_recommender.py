"""
Clothing Size Recommendation System
Author: manamendraJN
Date: 2025-11-18
"""

from config import Config

class SizeRecommender:
    """Recommend clothing sizes based on body measurements"""
    
    def __init__(self):
        self.size_charts = Config.SIZE_CHARTS
    
    def recommend_size(self, measurements, gender='male', region='us', garment_type='top'):
        """
        Recommend clothing size based on measurements
        
        Args:
            measurements: Dictionary of body measurements
            gender: 'male' or 'female'
            region: 'us', 'eu', 'uk', etc.
            garment_type: 'top', 'bottom', 'dress', etc.
        
        Returns:
            Size recommendation with confidence
        """
        gender = gender.lower()
        region = region.lower()
        
        if region not in self.size_charts:
            region = 'us'  # Default to US sizing
        
        if gender not in self.size_charts[region]:
            return {
                'error': f'Gender {gender} not found in size chart',
                'available_genders': list(self.size_charts[region].keys())
            }
        
        size_chart = self.size_charts[region][gender]
        
        # Get relevant measurements based on garment type
        if garment_type == 'top':
            size = self._match_top_size(measurements, size_chart, gender)
        elif garment_type == 'bottom':
            size = self._match_bottom_size(measurements, size_chart, gender)
        elif garment_type == 'dress':
            size = self._match_dress_size(measurements, size_chart, gender)
        else:
            size = self._match_generic_size(measurements, size_chart, gender)
        
        return size
    
    def _match_top_size(self, measurements, size_chart, gender):
        """Match size for tops (shirts, t-shirts, jackets)"""
        chest = measurements.get('chest')
        waist = measurements.get('waist')
        
        if not chest:
            return {'error': 'Chest measurement required for top sizing'}
        
        best_match = None
        best_score = float('inf')
        
        for size_name, size_ranges in size_chart.items():
            chest_range = size_ranges.get('chest', (0, 999))
            
            # Calculate how well the measurement fits
            chest_min, chest_max = chest_range
            chest_center = (chest_min + chest_max) / 2
            chest_score = abs(chest - chest_center)
            
            # Consider waist if available
            if waist and 'waist' in size_ranges:
                waist_range = size_ranges['waist']
                waist_min, waist_max = waist_range
                waist_center = (waist_min + waist_max) / 2
                waist_score = abs(waist - waist_center)
                
                total_score = chest_score + (waist_score * 0.5)  # Chest is more important
            else:
                total_score = chest_score
            
            if total_score < best_score:
                best_score = total_score
                best_match = size_name
        
        # Calculate confidence
        confidence = self._calculate_confidence(chest, size_chart[best_match]['chest'])
        
        return {
            'size': best_match.upper(),
            'confidence': confidence,
            'garment_type': 'top',
            'measurements_used': ['chest', 'waist'] if waist else ['chest'],
            'size_range': size_chart[best_match],
            'fit_advice': self._get_fit_advice(confidence)
        }
    
    def _match_bottom_size(self, measurements, size_chart, gender):
        """Match size for bottoms (pants, jeans, shorts)"""
        waist = measurements.get('waist')
        hip = measurements.get('hip')
        
        if not waist:
            return {'error': 'Waist measurement required for bottom sizing'}
        
        best_match = None
        best_score = float('inf')
        
        for size_name, size_ranges in size_chart.items():
            waist_range = size_ranges.get('waist', (0, 999))
            
            waist_min, waist_max = waist_range
            waist_center = (waist_min + waist_max) / 2
            waist_score = abs(waist - waist_center)
            
            # Consider hip if available (important for pants fit)
            if hip and 'hip' in size_ranges:
                hip_range = size_ranges['hip']
                hip_min, hip_max = hip_range
                hip_center = (hip_min + hip_max) / 2
                hip_score = abs(hip - hip_center)
                
                total_score = waist_score + hip_score
            else:
                total_score = waist_score
            
            if total_score < best_score:
                best_score = total_score
                best_match = size_name
        
        confidence = self._calculate_confidence(waist, size_chart[best_match]['waist'])
        
        return {
            'size': best_match.upper(),
            'confidence': confidence,
            'garment_type': 'bottom',
            'measurements_used': ['waist', 'hip'] if hip else ['waist'],
            'size_range': size_chart[best_match],
            'fit_advice': self._get_fit_advice(confidence),
            'inseam_recommendation': self._recommend_inseam(measurements)
        }
    
    def _match_dress_size(self, measurements, size_chart, gender):
        """Match size for dresses"""
        chest = measurements.get('chest')
        waist = measurements.get('waist')
        hip = measurements.get('hip')
        
        if not all([chest, waist, hip]):
            return {'error': 'Chest, waist, and hip measurements required for dress sizing'}
        
        best_match = None
        best_score = float('inf')
        
        for size_name, size_ranges in size_chart.items():
            scores = []
            
            for measurement_name in ['chest', 'waist', 'hip']:
                if measurement_name in size_ranges:
                    range_min, range_max = size_ranges[measurement_name]
                    center = (range_min + range_max) / 2
                    value = measurements[measurement_name]
                    scores.append(abs(value - center))
            
            total_score = sum(scores)
            
            if total_score < best_score:
                best_score = total_score
                best_match = size_name
        
        # Use chest for confidence calculation
        confidence = self._calculate_confidence(chest, size_chart[best_match]['chest'])
        
        return {
            'size': best_match.upper(),
            'confidence': confidence,
            'garment_type': 'dress',
            'measurements_used': ['chest', 'waist', 'hip'],
            'size_range': size_chart[best_match],
            'fit_advice': self._get_fit_advice(confidence)
        }
    
    def _match_generic_size(self, measurements, size_chart, gender):
        """Generic size matching using available measurements"""
        # Use chest as primary measurement
        return self._match_top_size(measurements, size_chart, gender)
    
    def _calculate_confidence(self, measurement, size_range):
        """Calculate confidence score (0-100)"""
        min_val, max_val = size_range
        
        # Check if measurement is within range
        if min_val <= measurement <= max_val:
            # Calculate how centered the measurement is
            center = (min_val + max_val) / 2
            range_size = max_val - min_val
            deviation = abs(measurement - center)
            
            # Confidence is higher when closer to center
            confidence = max(0, 100 - (deviation / range_size * 100))
            return round(confidence)
        else:
            # Measurement is outside range
            if measurement < min_val:
                distance = min_val - measurement
            else:
                distance = measurement - max_val
            
            # Reduce confidence based on distance
            confidence = max(0, 100 - (distance * 5))
            return round(confidence)
    
    def _get_fit_advice(self, confidence):
        """Get fit advice based on confidence"""
        if confidence >= 90:
            return "Perfect fit! This size should work great for you."
        elif confidence >= 75:
            return "Good fit! This size is recommended."
        elif confidence >= 60:
            return "Acceptable fit. Consider trying both this size and adjacent sizes."
        else:
            return "Uncertain fit. We recommend trying multiple sizes or getting custom measurements."
    
    def _recommend_inseam(self, measurements):
        """Recommend inseam length for pants"""
        leg_length = measurements.get('leg-length')
        height = measurements.get('height')
        
        if leg_length:
            # Convert cm to inches for inseam
            inseam_cm = leg_length * 0.85  # Approximate inseam from leg length
            inseam_inches = inseam_cm / 2.54
            
            return {
                'inseam_cm': round(inseam_cm, 1),
                'inseam_inches': round(inseam_inches, 1),
                'recommendation': self._get_inseam_category(inseam_inches)
            }
        elif height:
            # Estimate from height
            inseam_inches = (height / 2.54) * 0.47  # Rough approximation
            return {
                'inseam_inches': round(inseam_inches, 1),
                'recommendation': self._get_inseam_category(inseam_inches),
                'note': 'Estimated from height'
            }
        
        return None
    
    def _get_inseam_category(self, inseam_inches):
        """Get inseam category"""
        if inseam_inches < 28:
            return "Short (S)"
        elif inseam_inches < 32:
            return "Regular (R)"
        elif inseam_inches < 36:
            return "Long (L)"
        else:
            return "Extra Long (XL)"
    
    def get_all_sizes(self, measurements, gender='male', region='us'):
        """Get recommendations for all garment types"""
        return {
            'top': self.recommend_size(measurements, gender, region, 'top'),
            'bottom': self.recommend_size(measurements, gender, region, 'bottom'),
            'dress': self.recommend_size(measurements, gender, region, 'dress') if gender == 'female' else None
        }
    
    def compare_sizes(self, measurements, gender='male'):
        """Compare sizes across different regions"""
        regions = ['us']  # Can extend to 'eu', 'uk', etc.
        comparison = {}
        
        for region in regions:
            comparison[region] = self.recommend_size(measurements, gender, region, 'top')
        
        return comparison