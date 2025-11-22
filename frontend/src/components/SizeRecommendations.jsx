import React from 'react';
import { ShoppingBag, TrendingUp } from 'lucide-react';

const SizeRecommendations = ({ sizeData }) => {
  if (!sizeData) return null;

  const sizes = Object.entries(sizeData).filter(([_, data]) => data && !data.error);

  if (sizes.length === 0) return null;

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'bg-green-500';
    if (confidence >= 75) return 'bg-blue-500';
    if (confidence >= 60) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3 text-gray-800">
        <ShoppingBag className="w-7 h-7 text-purple-600" />
        Size Recommendations
      </h2>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sizes.map(([type, rec]) => (
          <div 
            key={type} 
            className="bg-gradient-to-br from-purple-50 to-pink-50 p-6 rounded-lg border-2 border-purple-200 hover:shadow-lg transition"
          >
            <p className="text-sm text-gray-600 uppercase tracking-wide mb-2 font-medium">
              {type}
            </p>
            
            <p className="text-5xl font-bold text-purple-600 mb-4">
              {rec.size}
            </p>

            {/* Confidence Bar */}
            <div className="mb-3">
              <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
                <span>Confidence</span>
                <span className="font-semibold">{rec.confidence}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div
                  className={`h-full ${getConfidenceColor(rec.confidence)} transition-all duration-500 rounded-full`}
                  style={{ width: `${rec.confidence}%` }}
                ></div>
              </div>
            </div>

            {/* Fit Advice */}
            <p className="text-sm text-gray-700 mb-3 flex items-start gap-2">
              <TrendingUp className="w-4 h-4 text-purple-600 flex-shrink-0 mt-0.5" />
              {rec.fit_advice}
            </p>

            {/* Size Range */}
            {rec.size_range && (
              <div className="text-xs text-gray-600 bg-white p-2 rounded">
                <p className="font-semibold mb-1">Size Range:</p>
                {Object.entries(rec.size_range).map(([measurement, range]) => (
                  <p key={measurement} className="capitalize">
                    {measurement}: {range[0]}-{range[1]} cm
                  </p>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default SizeRecommendations;