import React, { useRef } from "react";

type Props = {
  label: string;
  onFileSelected: (file: File | null) => void;
  selectedFile?: File | null;
};

export default function Dropzone({ label, onFileSelected, selectedFile }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0] ?? null;
    onFileSelected(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    onFileSelected(file);
  };

  return (
    <label
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      style={{
        display: "block",
        border: "2px dashed #888",
        padding: "20px",
        borderRadius: "12px",
        cursor: "pointer",
        marginBottom: "16px",
        background: "#fafafa",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: "6px" }}>{label}</div>
      <div style={{ fontSize: "0.9rem", color: "#666" }}>
        Drag and drop or click to select
      </div>
      <div style={{ marginTop: "8px", fontSize: "0.9rem" }}>
        {selectedFile ? `Selected: ${selectedFile.name}` : "No file selected"}
      </div>

      <input
        ref={inputRef}
        type="file"
        style={{ display: "none" }}
        onChange={handleChange}
      />
    </label>
  );
}
