import React, { ChangeEvent } from "react";


interface UrlInputProps {
    value: string;
    onChange: (event: ChangeEvent<HTMLInputElement>) => void;
    disabled: boolean;
  }
  
  export const UrlInputSection: React.FC<UrlInputProps> = ({
    value,
    onChange,
    disabled,
  }) => (
    <div className="mb-2">
      <label htmlFor="url-input" className="block text-sm font-medium mb-2">
        Enter URL:
      </label>
      <input
        id="url-input"
        type="text"
        value={value}
        onChange={onChange}
        placeholder="https://example.com"
        disabled={disabled}
        className={`w-full border rounded p-2 text-black ${
          disabled ? "bg-gray-100 cursor-not-allowed opacity-50" : ""
        }`}
      />
    </div>
  );

  export default UrlInputSection;