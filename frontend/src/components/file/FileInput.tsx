import React, { RefObject, ChangeEvent } from "react";


interface FileUploadProps {
  inputRef: RefObject<HTMLInputElement>;
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
  disabled: boolean;
  fileName?: string;
  onClear?: () => void;
}
  
export const FileUploadSection: React.FC<FileUploadProps> = ({
  inputRef,
  onChange,
  disabled,
  fileName,
  onClear,
}) => (
  <div className="mb-2 w-full">
    <label
      htmlFor="file-upload"
      className="block text-sm font-medium mb-2"
    >
      Upload File
    </label>

    <div className="flex items-center gap-2">
      <input
        id="file-upload"
        type="file"
        ref={inputRef}
        onChange={onChange}
        disabled={disabled}
        className={`flex-1 rounded border border-[hsl(var(--input))] bg-[hsl(var(--background))] p-2 text-sm ${
          disabled ? "cursor-not-allowed bg-[hsl(var(--muted))] opacity-50" : ""
        }`}
      />

      {fileName && !disabled && (
        <button
          type="button"
          onClick={onClear}
          className="text-sm text-[hsl(var(--primary))] hover:underline"
        >
          Remove
        </button>
      )}
    </div>

    {fileName && (
      <p className="mt-1 truncate text-sm text-[hsl(var(--muted-foreground))]">
        Selected: {fileName}
      </p>
    )}
  </div>
);


export default FileUploadSection;