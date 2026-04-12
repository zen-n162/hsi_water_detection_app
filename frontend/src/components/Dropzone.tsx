import React, { useRef } from "react";

type Props = {
  label: string;
  onFileSelected: (file: File | null) => void;
};

export default function Dropzone({ label, onFileSelected }: Props) {
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
        padding: "24px",
        borderRadius: "12px",
        cursor: "pointer",
        marginBottom: "16px",
      }}
    >
      <div>{label}</div>
      <div style={{ fontSize: "0.9rem", color: "#666" }}>
        Drag and drop or click to select
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
