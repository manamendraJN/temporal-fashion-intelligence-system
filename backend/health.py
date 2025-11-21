"""
Health Insights and BMI Calculator
Author: manamendraJN
Date: 2025-11-18
"""

import numpy as np
from config import Config

class HealthAnalyzer:
    """Analyze body measurements and provide health insights"""
    
    def __init__(self):
        self.bmi_categories = Config.BMI_CATEGORIES
    
    def calculate_bmi(self, height_cm, weight_kg):
        """Calculate BMI from height and weight"""
        height_m = height_cm / 100.0
        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)
    
    def get_bmi_category(self, bmi):
        """Get BMI category"""
        for category, (min_val, max_val) in self.bmi_categories.items():
            if min_val <= bmi < max_val:
                return category
        return 'unknown'
    
    def get_bmi_insights(self, bmi, category):
        """Get health insights based on BMI"""
        insights = {
            'underweight': {
                'status': '⚠️ Underweight',
                'description': 'Your BMI indicates you may be underweight.',
                'recommendations': [
                    'Consult a healthcare provider for personalized advice',
                    'Focus on nutrient-dense foods',
                    'Consider strength training exercises',
                    'Monitor your health regularly'
                ],
                'health_risks': [
                    'Weakened immune system',
                    'Nutritional deficiencies',
                    'Decreased bone density'
                ]
            },
            'normal': {
                'status': '✅ Normal Weight',
                'description': 'Your BMI is in the healthy range. Great job!',
                'recommendations': [
                    'Maintain a balanced diet',
                    'Stay physically active (150+ min/week)',
                    'Keep consistent sleep schedule',
                    'Regular health check-ups'
                ],
                'health_risks': []
            },
            'overweight': {
                'status': '⚠️ Overweight',
                'description': 'Your BMI indicates you may be overweight.',
                'recommendations': [
                    'Consult a healthcare provider',
                    'Adopt a balanced, calorie-controlled diet',
                    'Increase physical activity gradually',
                    'Focus on sustainable lifestyle changes'
                ],
                'health_risks': [
                    'Increased risk of type 2 diabetes',
                    'Higher blood pressure risk',
                    'Elevated cholesterol levels'
                ]
            },
            'obese': {
                'status': '🚨 Obese',
                'description': 'Your BMI indicates obesity. Medical consultation recommended.',
                'recommendations': [
                    'Consult a healthcare provider immediately',
                    'Work with a nutritionist for meal planning',
                    'Start with low-impact exercises',
                    'Consider medical weight management programs'
                ],
                'health_risks': [
                    'Type 2 diabetes',
                    'Heart disease',
                    'High blood pressure',
                    'Sleep apnea',
                    'Joint problems'
                ]
            }
        }
        
        return insights.get(category, {
            'status': 'Unknown',
            'description': 'Unable to determine health status',
            'recommendations': ['Consult a healthcare provider'],
            'health_risks': []
        })
    
    def analyze_body_proportions(self, measurements):
        """Analyze body proportions and ratios"""
        results = {}
        
        # Waist-to-Hip Ratio (WHR)
        if 'waist' in measurements and 'hip' in measurements:
            whr = measurements['waist'] / measurements['hip']
            results['waist_to_hip_ratio'] = {
                'value': round(whr, 3),
                'category': self._get_whr_category(whr),
                'description': self._get_whr_description(whr)
            }
        
        # Chest-to-Waist Ratio
        if 'chest' in measurements and 'waist' in measurements:
            cwr = measurements['chest'] / measurements['waist']
            results['chest_to_waist_ratio'] = {
                'value': round(cwr, 3),
                'description': 'Athletic build' if cwr > 1.1 else 'Normal build'
            }
        
        # Body symmetry
        if 'shoulder-breadth' in measurements and 'hip' in measurements:
            shr = measurements['shoulder-breadth'] / measurements['hip']
            results['shoulder_to_hip_ratio'] = {
                'value': round(shr, 3),
                'description': self._get_body_shape(shr)
            }
        
        return results
    
    def _get_whr_category(self, whr):
        """Get WHR category (health risk assessment)"""
        # General guidelines (varies by gender)
        if whr < 0.85:
            return 'Low risk'
        elif whr < 0.90:
            return 'Moderate risk'
        else:
            return 'High risk'
    
    def _get_whr_description(self, whr):
        """Get WHR description"""
        if whr < 0.85:
            return 'Your waist-to-hip ratio is in the healthy range'
        elif whr < 0.90:
            return 'Consider monitoring your waist circumference'
        else:
            return 'Higher cardiovascular risk - consult healthcare provider'
    
    def _get_body_shape(self, shr):
        """Determine body shape from shoulder-to-hip ratio"""
        if shr > 1.1:
            return 'V-shaped (athletic)'
        elif shr > 0.95:
            return 'Rectangular'
        else:
            return 'A-shaped (pear)'
    
    def generate_fitness_goals(self, measurements, bmi_category):
        """Generate personalized fitness goals"""
        goals = []
        
        if bmi_category == 'underweight':
            goals = [
                'Gain 0.5-1 kg per week through strength training',
                'Increase caloric intake by 300-500 calories/day',
                'Focus on compound exercises (squats, deadlifts)',
                'Consume 1.6-2.2g protein per kg body weight'
            ]
        elif bmi_category == 'normal':
            goals = [
                'Maintain current weight',
                'Build lean muscle mass',
                'Improve cardiovascular endurance',
                'Enhance flexibility and mobility'
            ]
        elif bmi_category in ['overweight', 'obese']:
            goals = [
                'Lose 0.5-1 kg per week safely',
                'Reduce caloric intake by 500 calories/day',
                'Aim for 150+ minutes of moderate exercise weekly',
                'Focus on whole foods and portion control'
            ]
        
        return goals
    
    def get_full_health_report(self, measurements, weight_kg=None, gender='male'):
        """Generate comprehensive health report"""
        report = {
            'measurements': measurements,
            'timestamp': str(np.datetime64('now'))
        }
        
        # BMI Analysis
        if 'height' in measurements and weight_kg:
            bmi = self.calculate_bmi(measurements['height'], weight_kg)
            bmi_category = self.get_bmi_category(bmi)
            bmi_insights = self.get_bmi_insights(bmi, bmi_category)
            
            report['bmi'] = {
                'value': bmi,
                'category': bmi_category,
                'status': bmi_insights['status'],
                'description': bmi_insights['description'],
                'recommendations': bmi_insights['recommendations'],
                'health_risks': bmi_insights['health_risks']
            }
        
        # Body Proportions
        report['proportions'] = self.analyze_body_proportions(measurements)
        
        # Fitness Goals
        if 'bmi' in report:
            report['fitness_goals'] = self.generate_fitness_goals(
                measurements, 
                report['bmi']['category']
            )
        
        # General Health Tips
        report['health_tips'] = [
            '💧 Drink at least 2-3 liters of water daily',
            '🥗 Eat a balanced diet with fruits and vegetables',
            '😴 Get 7-9 hours of quality sleep',
            '🏃 Stay physically active (at least 30 min/day)',
            '🧘 Manage stress through meditation or yoga',
            '🩺 Get regular health check-ups'
        ]
        
        return report