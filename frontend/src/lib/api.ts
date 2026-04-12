export async function runInference(formData: FormData) {
  const res = await fetch("http://127.0.0.1:8000/inference/run", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Inference API failed: ${res.status}`);
  }
  return res.json();
}

export async function getGrayscalePreview(formData: FormData) {
  const res = await fetch("http://127.0.0.1:8000/preview/grayscale", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`Preview API failed: ${res.status}`);
  }
  return res.json();
}
