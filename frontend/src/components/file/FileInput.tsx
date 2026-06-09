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
        className={`flex-1 border rounded p-2 ${
          disabled ? "bg-gray-100 cursor-not-allowed opacity-50" : ""
        }`}
      />

      {fileName && !disabled && (
        <button
          type="button"
          onClick={onClear}
          className="text-sm text-red-500 hover:underline"
        >
          Remove
        </button>
      )}
    </div>

    {fileName && (
      <p className="mt-1 text-sm text-slate-400 truncate">
        Selected: {fileName}
      </p>
    )}
  </div>
);


export default FileUploadSection;