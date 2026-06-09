import React, { ChangeEvent } from "react";

interface TextInputProps {
    value: string;
    onChange: (event: ChangeEvent<HTMLTextAreaElement>) => void;
    disabled: boolean;
  }
  
  export const TextInputSection: React.FC<TextInputProps> = ({
    value,
    onChange,
    disabled,
  }) => (
    <div className="mb-4">
      <label htmlFor="text-input" className="block text-sm font-medium mb-2">
        Paste Text:
      </label>
      <textarea
        id="text-input"
        rows={4}
        value={value}
        onChange={onChange}
        placeholder="Paste your content here..."
        disabled={disabled}
        className={`w-full border rounded p-2 resize-y  text-black ${
          disabled ? "bg-gray-100 cursor-not-allowed opacity-50" : ""
        }`}
      />
    </div>
  );

export default TextInputSection;