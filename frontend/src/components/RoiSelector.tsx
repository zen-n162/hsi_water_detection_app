import { useEffect, useRef, useState } from "react";
import ReactCrop, { type Crop, type PixelCrop } from "react-image-crop";
import "react-image-crop/dist/ReactCrop.css";

type RoiSelectorProps = {
  imageUrl?: string;
  imageWidth?: number;
  imageHeight?: number;
  onApply: (roi: {
    row_start: number;
    row_stop: number;
    col_start: number;
    col_stop: number;
  }) => void;
};

export default function RoiSelector({
  imageUrl,
  imageWidth,
  imageHeight,
  onApply,
}: RoiSelectorProps) {
  const [crop, setCrop] = useState<Crop>({
    unit: "px",
    x: 10,
    y: 10,
    width: 128,
    height: 128,
  });
  const [completedCrop, setCompletedCrop] = useState<PixelCrop | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    setCompletedCrop(null);
  }, [imageUrl]);

  const handleApply = () => {
    if (!completedCrop || !imgRef.current || !imageWidth || !imageHeight) return;

    const displayedWidth = imgRef.current.width;
    const displayedHeight = imgRef.current.height;

    const scaleX = imageWidth / displayedWidth;
    const scaleY = imageHeight / displayedHeight;

    const col_start = Math.max(0, Math.floor(completedCrop.x * scaleX));
    const col_stop = Math.min(imageWidth, Math.ceil((completedCrop.x + completedCrop.width) * scaleX));
    const row_start = Math.max(0, Math.floor(completedCrop.y * scaleY));
    const row_stop = Math.min(imageHeight, Math.ceil((completedCrop.y + completedCrop.height) * scaleY));

    onApply({ row_start, row_stop, col_start, col_stop });
  };

  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12, background: "#111" }}>
      <h3 style={{ marginTop: 0 }}>ROI Selection</h3>

      {!imageUrl ? (
        <div style={{ color: "#bbb", padding: "24px 0" }}>
          Preview grayscale image will appear here.
        </div>
      ) : (
        <>
          <ReactCrop
            crop={crop}
            onChange={(c) => setCrop(c)}
            onComplete={(c) => setCompletedCrop(c)}
          >
            <img
              ref={imgRef}
              src={imageUrl}
              alt="grayscale preview"
              style={{ maxWidth: "100%", display: "block" }}
            />
          </ReactCrop>

          <button
            onClick={handleApply}
            style={{ marginTop: 12, padding: "8px 14px" }}
          >
            Apply ROI
          </button>

          {completedCrop && (
            <div style={{ marginTop: 8, fontSize: "0.9rem", color: "#ddd" }}>
              crop x={Math.round(completedCrop.x)} y={Math.round(completedCrop.y)} w={Math.round(completedCrop.width)} h={Math.round(completedCrop.height)}
            </div>
          )}
        </>
      )}
    </div>
  );
}
