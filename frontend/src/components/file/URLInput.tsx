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
        className={`w-full rounded border border-[hsl(var(--input))] bg-[hsl(var(--background))] p-2 text-sm text-[hsl(var(--foreground))] ${
          disabled ? "cursor-not-allowed bg-[hsl(var(--muted))] opacity-50" : ""
        }`}
      />
    </div>
  );

  export default UrlInputSection;