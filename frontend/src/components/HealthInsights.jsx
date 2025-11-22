import React from 'react';
import { Activity, Heart, TrendingUp, AlertCircle } from 'lucide-react';

const HealthInsights = ({ healthData }) => {
  if (!healthData) return null;

  const { bmi, proportions } = healthData;

  const getBMIColor = (category) => {
    const colors = {
      underweight: 'bg-yellow-100 border-yellow-300 text-yellow-800',
      normal: 'bg-green-100 border-green-300 text-green-800',
      overweight: 'bg-orange-100 border-orange-300 text-orange-800',
      obese: 'bg-red-100 border-red-300 text-red-800',
    };
    return colors[category] || 'bg-gray-100 border-gray-300 text-gray-800';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3 text-gray-800">
        <Activity className="w-7 h-7 text-green-600" />
        Health Insights
      </h2>

      {/* BMI Section */}
      {bmi && (
        <div className={`p-6 rounded-lg border-2 mb-6 ${getBMIColor(bmi.category)}`}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-medium opacity-80">Body Mass Index (BMI)</p>
              <p className="text-4xl font-bold mt-1">{bmi.value}</p>
            </div>
            <Heart className="w-12 h-12 opacity-50" />
          </div>
          
          <p className="text-lg font-semibold mb-2">{bmi.status}</p>
          <p className="text-sm opacity-90">{bmi.description}</p>

          {/* Recommendations */}
          {bmi.recommendations && bmi.recommendations.length > 0 && (
            <div className="mt-4 pt-4 border-t border-current opacity-80">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <TrendingUp className="w-4 h-4" />
                Recommendations:
              </p>
              <ul className="space-y-1 text-sm ml-6">
                {bmi.recommendations.slice(0, 3).map((rec, idx) => (
                  <li key={idx} className="list-disc">{rec}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Health Risks */}
          {bmi.health_risks && bmi.health_risks.length > 0 && (
            <div className="mt-4 pt-4 border-t border-current opacity-80">
              <p className="font-semibold mb-2 flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Health Considerations:
              </p>
              <ul className="space-y-1 text-sm ml-6">
                {bmi.health_risks.slice(0, 3).map((risk, idx) => (
                  <li key={idx} className="list-disc">{risk}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Body Proportions */}
      {proportions && Object.keys(proportions).length > 0 && (
        <div className="grid md:grid-cols-2 gap-4">
          {Object.entries(proportions).map(([key, data]) => (
            <div key={key} className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <p className="text-xs text-gray-600 uppercase tracking-wide mb-1">
                {key.replace(/_/g, ' ')}
              </p>
              <p className="text-xl font-bold text-blue-700 mb-1">
                {data.value?.toFixed(3)}
              </p>
              <p className="text-sm text-gray-700">{data.description || data.category}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default HealthInsights;