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
        className={`w-full resize-y rounded border border-[hsl(var(--input))] bg-[hsl(var(--background))] p-2 text-sm text-[hsl(var(--foreground))] ${
          disabled ? "cursor-not-allowed bg-[hsl(var(--muted))] opacity-50" : ""
        }`}
      />
    </div>
  );

export default TextInputSection;