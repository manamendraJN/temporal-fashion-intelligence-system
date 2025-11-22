import React from 'react';
import { Ruler } from 'lucide-react';

const MeasurementsDisplay = ({ measurements }) => {
  if (!measurements) return null;

  const measurementEntries = Object.entries(measurements);

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3 text-gray-800">
        <Ruler className="w-7 h-7 text-primary-600" />
        Body Measurements
      </h2>
      
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {measurementEntries.map(([key, data]) => (
          <div 
            key={key} 
            className="bg-gradient-to-br from-primary-50 to-primary-100 p-5 rounded-lg border border-primary-200 hover:shadow-md transition"
          >
            <p className="text-xs text-gray-600 uppercase tracking-wide mb-1">
              {key.replace('-', ' ')}
            </p>
            <p className="text-2xl font-bold text-primary-700">
              {data.display || `${data.value?.toFixed(1)} cm`}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MeasurementsDisplay;