import React from 'react';
import { Upload, X } from 'lucide-react';

const ImageUpload = ({ label, image, preview, onChange, onRemove }) => {
  const inputId = `upload-${label.toLowerCase().replace(' ', '-')}`;

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        {label}
      </label>
      
      <div className="relative border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors bg-white">
        {preview ? (
          <div className="relative">
            <img 
              src={preview} 
              alt={`${label} preview`} 
              className="max-h-64 mx-auto rounded-lg shadow-md"
            />
            <button
              type="button"
              onClick={onRemove}
              className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition shadow-lg"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 mb-4">Click to upload or drag and drop</p>
          </>
        )}
        
        <input
          type="file"
          accept="image/*"
          onChange={onChange}
          className="hidden"
          id={inputId}
        />
        
        {!preview && (
          <label
            htmlFor={inputId}
            className="cursor-pointer bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition inline-block font-medium"
          >
            Choose Image
          </label>
        )}
      </div>
    </div>
  );
};

export default ImageUpload;